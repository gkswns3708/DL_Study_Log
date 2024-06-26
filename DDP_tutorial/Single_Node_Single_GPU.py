import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank (_type_): Unique identifier of each process
        world_size (_type_): Total number of process
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1235"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        # self.model = model.to(gpu_id) # Original code(Single Node & Single GPU)
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize : {b_sz} : Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        # From Trainer we wrapped the self.model object with DDP
        # If we want to acess the each Model Parameters, we have to call model.module
        # ckp = self.model.state_dict() -> Orignal(Single Node & Single GPU)
        ckp = self.model.module.state_dict()
        PATH = f"./save/single_gpu_checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        # 기존의 방식대로 한다면 불필요한 rebundancy가 만들어질 것이라고 함.
        # 모두 동일한 replica를 사용중이니 save 과정은 1개의 GPU에서만 발생하면 됨.
        # 그래서 if 문 조건에 gpu_id 조건을 추가함.
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
    
def load_train_objs():
    train_set = MyTrainDataset(2048) # Load my Dataset
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer
    
def prepare_dataloader(dataset: Dataset, batch_size:int):
    # DDP를 위해서는 DataLoader에서 사용할 Sampler로 Distributed Sampler를 사용해야함.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, # set_epoch를 통해 random shuffle이 보장됨.
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    

    # 모든 distributed group을 생성함.
    # 여기서 mp.spawn이 자동적으로 rank르 생성해줌.
    # nprocs : refers to a number of processes. 아래의 예시에서는 world_size를 했는데, 이것은 Single Node & Multi GPU 상황이기 때문.
    world_size = torch.cuda.device_count()
    print(world_size, "-world_size")
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    