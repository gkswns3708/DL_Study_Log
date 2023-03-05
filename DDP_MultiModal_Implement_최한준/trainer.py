import os
import time
from glob import glob

import utils
from utils import AverageMeter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,  # for example save_every
        args,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"]) if args.train else 'cuda:0'
        self.global_rank = int(os.environ["RANK"]) if args.train else 'cuda:0'
        self.model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=True) if args.train  else model
        self.dataloader = dataloader,
        self.dataloader = self.dataloader[0]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = 9999
        self.metric = 0
        self.loss_fn = nn.CrossEntropyLoss().to(self.local_rank)

    def _run_epoch(self, epoch, train=True):
        batch_time = AverageMeter()     
        data_time = AverageMeter()      
        losses = AverageMeter()         
        accuracies = AverageMeter()
        f1_accuracies = AverageMeter()
        if train:
            self.model = self.model.train()
        sent_count = AverageMeter()   
        start = end = time.time()
        correct_predictions = 0
        b_sz = next(iter(self.dataloader))['input_ids'].size(0) 
        print(f"[GPU: {self.local_rank}], [Epoch : {epoch} | B_sz : {b_sz}],  [Required Steps: {len(self.dataloader)}]")
        for step, d in enumerate(self.dataloader):
            input_ids = d['input_ids'].to(self.local_rank)
            attention_mask = d["attention_mask"].to(self.local_rank)
            pixel_values = d["pixel_values"].to(self.local_rank)
            cats1 = d["cats1"].to(self.local_rank)
            cats2 = d["cats2"].to(self.local_rank)
            cats3 = d["cats3"].to(self.local_rank)

            outputs1, outputs2, outputs3 = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            _, preds = torch.max(outputs3, dim=1)

            loss1 = self.loss_fn(outputs1, cats1)
            loss2 = self.loss_fn(outputs2, cats2)
            loss3 = self.loss_fn(outputs3, cats3)

            total_loss = loss1 * 0.1 + loss2 * 0.1 + loss3 * 0.8

            correct_predictions += torch.sum(preds == cats3)
            losses.update(total_loss.item())
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            sent_count.update(b_sz)
            if step % 200 == 0 or step == (len(self.dataloader)-1):
                        acc,f1_acc = utils.calc_tour_acc(outputs3, cats3)
                        accuracies.update(acc, b_sz)
                        f1_accuracies.update(f1_acc, b_sz)
            
                        print('Epoch: [{0}][{1}/{2}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Elapsed {remain:s} '
                            'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                            'Acc: {acc.val:.3f}({acc.avg:.3f}) '   
                            'f1_Acc: {f1_acc.val:.3f}({f1_acc.avg:.3f}) '           
                            'sent/s {sent_s:.0f} '
                            .format(
                            epoch, step+1, len(self.dataloader),
                            data_time=data_time, loss=losses,
                            acc=accuracies,
                            f1_acc=f1_accuracies,
                            remain=utils.timeSince(start, float(step+1)/len(self.dataloader)),
                            sent_s=sent_count.avg/batch_time.avg
                            ))
        
        return correct_predictions.double() / len(self.dataloader), losses.avg
    
    def train(self, max_epochs: int):
        max_loss = 999999
        for epoch in range(1, max_epochs + 1):
            acc, loss = self._run_epoch(epoch)
            if loss < max_loss:
                ckp = self.model.module.state_dict()
                torch.save(ckp, f"./save/best_loss_{loss}.pt")
                ckp_list = sorted(glob(os.path.join("./save", "*")), key=lambda x : float(x.split('_')[-1][:-3]))
                if len(ckp_list) > 3:
                    os.remove(ckp_list[-1])

    def inference(self):
        model_path = sorted(glob(os.path.join('./save', '*')))[0]
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.local_rank)
        self.model.eval()
        preds_arr1 = []
        preds_arr2 = []
        preds_arr3 = []
        for d in tqdm(self.dataloader):
            with torch.no_grad():
                input_ids = d['input_ids'].to(self.local_rank)
                attention_mask = d['attention_mask'].to(self.local_rank)
                pixel_values = d['pixel_values'].to(self.local_rank)

                outputs1, outputs2, outputs3 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )

                _, pred1 = torch.max(outputs1,dim=1)
                _, pred2 = torch.max(outputs2,dim=1)
                _, pred3 = torch.max(outputs3,dim=1)

                preds_arr1.extend(pred1)
                preds_arr2.extend(pred2)
                preds_arr3.extend(pred3)

        
        return preds_arr1, preds_arr2, preds_arr3