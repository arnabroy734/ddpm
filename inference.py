from noise_scheduler import LinearNoiseScheduler
import torch
import yaml
from pathlib import Path
from architecture import Unet

class Inference:
    def __init__(self, config):
        # load the lateest model
        modelfolder = Path.cwd()/config['train_params']['modelpath']
        self.model = Unet(config['model_params'])
        try:
            checkpoint = torch.load(modelfolder/config['train_params']['checkpoint_latest'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint['epoch']} epoch checkpoint ..")
        except Exception as e:
            raise ValueError("No checkpoint found ..")
        
        # schdeuler
        self.noise_scheduler = LinearNoiseScheduler(
            beta_min = config['noise_params']['beta_min'],
            beta_max=config['noise_params']['beta_max'],
            T=config['noise_params']['timesteps']
        )
        self.T = config['noise_params']['timesteps']

        # image dim
        self.batch = config['data_params']['batch']
        self.h = config['model_params']['im_height']
        self.w = config['model_params']['im_width']
        self.channel = config['model_params']['im_channels']


    def generate(self):
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        xt = torch.randn(size=(self.batch, self.channel, self.h, self.w)).to(device)
        self.model.to(device)

        with torch.no_grad():
            for i in reversed(range(self.T)):
                t = torch.full((self.batch, ), i).to(device)
                noise_pred = self.model(xt, t)
                if torch.sum(torch.isnan(noise_pred)) > 0:
                    print(f"Noise prediction NaN at {i+1} th timestep")
                    return xt
                    
                xt, x0 = self.noise_scheduler.sample(xt, i, noise_pred)
                if torch.sum(torch.isnan(xt)) > 0:  
                    print(f'NaN at {i+1} th timestep: xt')
                    break
                if torch.sum(torch.isnan(x0)):
                    print(f'NaN at {i+1} th timestep: x0')
                    break
                if (i+1)%100 == 0:
                    print(f"{i+1} iteration complete")
        return xt
        

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        f.close()
    inference = Inference(config)
    inference.generate()