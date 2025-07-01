from architecture import Unet
from dataset import CelebDataset, prepare_train_data
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from noise_scheduler import LinearNoiseScheduler
import numpy as np

# DDP Imports
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group
import os

# DDP Initialise Process Group
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_dataloader(batch: int, height: int, width: int, trainpath: str):
    data = CelebDataset(height, width, trainpath)
    loader = DataLoader(data, batch_size=batch, sampler=DistributedSampler(data))
    return loader

class Trainer:
    def __init__(self, config, gpu_id): # gpu_id is for DDP
        # Get the dataloader
        self.loader = get_dataloader(
            batch=config['data_params']['batch'],
            height=config['model_params']['im_height'],
            width=config['model_params']['im_width'],
            trainpath=config['data_params']['trainpath']

        )
        print(f"Data loaded successfully ..\n")

        # Get the model
        self.modelfolder = Path.cwd()/config['train_params']['modelpath']
        Path.mkdir(self.modelfolder, exist_ok=True)
        self.model = Unet(config['model_params'])
        self.optimiser = torch.optim.Adam(params=self.model.parameters(), lr=config['train_params']['lr'])
        try:
            checkpoint = torch.load(self.modelfolder/config['train_params']['checkpoint_latest'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model and optimiser loaded from {checkpoint['epoch']} epoch checkpoint ..")
        except Exception as e:
            print(f"New model initialised ..")

        # Wrap the model in DDP
        self.device_id = gpu_id
        self.model.to(f'cuda:{gpu_id}')
        self.model = DDP(module=self.model, device_ids=[gpu_id])
        
        # Noise-Scheduler
        self.noise_scheduler = LinearNoiseScheduler(
            beta_min = config['noise_params']['beta_min'],
            beta_max=config['noise_params']['beta_max'],
            T=config['noise_params']['timesteps']
        )
        self.T = config['noise_params']['timesteps']

        # Other params
        self.epochs = config['train_params']['epochs']
        self.checkpoint_epochs = config['train_params']['checkpoint_epoch']
        self.latest_modelfile = self.modelfolder/config['train_params']['checkpoint_latest']
        

        # loss function
        self.criterion = torch.nn.MSELoss()


    def train(self):
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        device = f'cuda:{self.device_id}'
        # self.model.to(device)

        for state in self.optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        for epoch in range(self.epochs):
            epoch_loss = []
            for i,X in enumerate(self.loader):
                X = X.to(device)
                t = torch.randint(low=0, high=self.T, size=(X.shape[0], )).to(device)
                noise_sample = torch.randn(size=X.shape).to(device)

                noisy_X = self.noise_scheduler.add_noise(X, t, noise_sample)
                noise_estimate = self.model(noisy_X, t)
                loss = self.criterion(noise_sample, noise_estimate)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if (i+1)%1 == 0:
                    print(f"At {i+1}th step the loss is {loss.item()}")
                epoch_loss.append(loss.item())
            epoch_loss = np.mean(epoch_loss)
            print(f"\nEnd of epoch {epoch+1}: loss = {epoch_loss}")

            if (epoch+1) % self.checkpoint_epochs == 0 and self.device_id == 0: # Device ID check is required for DDP
                # save the model
                torch.save({
                    'model_state_dict' : self.model.module.state_dict(), # type: ignore
                    'optimizer_state_dict' : self.optimiser.state_dict(),
                    'epoch' : epoch+1
                }, self.latest_modelfile)


def main(rank: int, world_size: int, config: dict):
    ddp_setup(rank, world_size)
    trainer = Trainer(config, gpu_id=rank)
    trainer.train()
    destroy_process_group()



if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        f.close()
    prepare_train_data(
        datapath=config['data_params']['datapath'],
        trainpath=config['data_params']['trainpath'],
        num_train_sample=config['train_params']['num_training_samples']
    )
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config), nprocs=world_size) # type: ignore
    