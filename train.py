from architecture import Unet
from dataset import CelebDataset
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from noise_scheduler import LinearNoiseScheduler
import numpy as np

def get_dataloader(datapath: str, batch: int):
    data = CelebDataset(datapath)
    loader = DataLoader(data, batch_size=batch)
    return loader

class Trainer:
    def __init__(self, config):
        # Get the dataloader
        self.loader = get_dataloader(
            datapath=config['data_params']['datapath'],
            batch=config['data_params']['batch']
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
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

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

            if (epoch+1) % self.checkpoint_epochs == 0:
                # save the model
                torch.save({
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimiser.state_dict(),
                    'epoch' : epoch+1
                }, self.latest_modelfile)



if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        f.close()
    trainer = Trainer(config)
    trainer.train()
    