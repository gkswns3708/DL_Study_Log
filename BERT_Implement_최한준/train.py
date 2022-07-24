import os, math, argparse, collections
import random
from tqdm import trange, tqdm

import numpy as np
import wandb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from Data.vocab import load_vocab
import config as cfg
import model as bert
import data
import optimization as optim


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO : Making Vocab Process
""" random seed """


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def eval_epoch(config, rank, model, data_loader):
    matchs = []
    model.eval()

    n_word_total = 0
    n_correct_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid({rank})") as pbar:
        for i, value in enumerate(data_loader):
            labels, inputs, segments = map(
                lambda v: v.type(torch.LongTensor).to(config.device), value
            )

            outputs = model(inputs, segments)
            logits_cls = outputs[0]
            _, indices = logits_cls.max(1)  # TODO: 이거 dim=1이라는 뜻인가?

            match = torch.eq(indices, labels).detach().cpu()
            matchs.extend(match)
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_posefix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0


def train_epoch(
    config, rank, epoch, model, criterion_cls, optimizer, scheduler, train_loader
):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, inputs, segments = map(
                lambda v: v.type(torch.LongTensor).to(config.device), value
            )

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls = outputs[0]

            loss_cls = criterion_cls(logits_cls, labels)
            loss = loss_cls

            loss_val = loss_cls.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.steop()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")

    return np.mean(losses)


def train_model(rank, world_size, args):
    master = world_size == 0 or rank % world_size == 0
    if master:
        wandb.init(project="BERT Pre-train & Fine-tuing")

    vocab = load_vocab(args.vocab)

    config = cfg.Config.load(args.config)
    config.n_enc_vocab = len(vocab)
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config)

    best_epoch, best_loss, best_score = 0, 0, 0
    model = bert.MovieClassification(config)
    if os.path.isfile(args.save):
        best_epoch, best_loss, best_score = model.load(args.save)
        print(f"rank : {rank}, load_state_dict from : {args.save}")
    elif os.path.isfile(args.pretrain):
        epoch, loss = model.bert.load(args.pretrain)
        print(
            f"rank : {rank}, load_state_dict from : {args.save}, epoch = {epoch}, loss = {loss}"
        )

    if 1 < args.n_gpu:
        model.to(config.device)
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
    else:
        model.to(config.device)
    if master:
        wandb.watch(model)  # TODO:wandb watch api 뜻 이해하기.

    # TODO: Loss Function, Optimizer 등과 같은 Hyper Parameter가 실제 구현과 동일한지 비교하기
    criterion_cls = torch.nn.CrossEntropyLoss()

    train_loader, train_sampler = data.build_data_loader(
        vocab, args.train, args, shuffle=True
    )
    test_loader, _ = data.build_data_loader(vocab, args.test, args, shuffle=False)

    t_total = len(train_loader) * args.epoch
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                parameter_value
                for parameter_name, parameter_value in model.named_parameters()
                if not any(nd in parameter_name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter_value
                for parameter_name, parameter_value in model.named_parameters()
                if any(nd in parameter_name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = optim.get_linear_scehdule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    offset = best_epoch
    for step in trange(args.epoch, desc="Epoch"):
        if train_sampler:
            train_sampler.set_epoch(step)
        epoch = step + offset

        loss = train_epoch(
            config,
            rank,
            epoch,
            model,
            criterion_cls,
            optimizer,
            scheduler,
            train_loader,
        )
        score = eval_epoch(config, rank, model, test_loader)

        if master:
            wandb.log(
                {"loss": loss, "accuracy": score}
            )  # TODO: F_1 score Metric을 추가해도 될 듯 하다.

        if master and best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
            if isinstance(model, DistributedDataParallel):
                model.module.save(best_epoch, best_loss, best_score, args.save)
            else:
                model.save(best_epoch, best_loss, best_score, args.save)
            print(
                f">>>> rank: {rank} save model to {args.save}, epoch={best_epoch}, loss={best_loss:.3f}, socre={best_score:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        type=str,
        required=False,
        help="Config File Path",
    )
    parser.add_argument(
        "--vocab",
        default="./Data/kowiki.model",
        type=str,
        required=False,
        help="Vocab File Path",
    )
    parser.add_argument(
        "--train",
        default="./Data/ratings_train.json",
        type=str,
        required=False,
        help="input train file",
    )
    parser.add_argument(
        "--test",
        default="./Data/ratings_test.json",
        type=str,
        required=False,
        help="input test file",
    )
    parser.add_argument(
        "--save_path",
        default="./save/save_best.pth",
        type=str,
        required=False,
        help="Path for Saving best Performance model",
    )
    parser.add_argument(
        "--pretrain",
        default="./save/save_pretrain.pth",
        type=str,
        required=False,
        help="Path for pre-trained weight",
    )
    parser.add_argument("--epoch", default=20, type=int, required=False, help="Epoch")
    parser.add_argument(
        "--batch", default=16, type=int, required=False, help="Batch Size"
    )
    parser.add_argument(
        "--gpu", default=None, type=int, required=False, help="GPU id to use."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed for Initialization",
    )
    parser.add_argument(
        "--weight_decay", default=0, type=float, required=False, help="Weight Decay"
    )  # TODO: Weight Decay 개념 및 Defalut BERT Weight Decay 공부. 아래의 것들도 마찬가지
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        required=False,
        help="Learning Rate",
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, required=False, help="Adam Epsilon"
    )
    parser.add_argument(
        "--warmup_steps", type=float, default=0, required=False, help="Warmup Steps"
    )

    args = parser.parse_args()

    args.n_gpu = 0
    args.gpu = 0
    set_seed(args)
    train_model(0, args.n_gpu, args)
