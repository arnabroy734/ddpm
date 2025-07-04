from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import shutil

def prepare_train_data(datapath, trainpath, num_train_sample):
    trainpath = Path.cwd()/trainpath
    if trainpath.exists() is False:
        trainpath.mkdir()
        folderpath = Path.cwd()/datapath
        all_files = []
        for i, file in enumerate(folderpath.iterdir()):
            all_files.append(file)
        np.random.shuffle(all_files)
        for i in range(min(num_train_sample, len(all_files))):
            shutil.copy(all_files[i], trainpath/all_files[i].name)
        print(f"{num_train_sample} samples copied to {trainpath}")
    else:
        print(f"Train path {trainpath} already exists. No samples copied.")
            

class CelebDataset(Dataset):
    def __init__(self,  height, width, trainpath):
        super().__init__()
        trainpath = Path.cwd()/trainpath
        self.images = []

        # If train path does not exist then copy random 'num_train_sample' samples inside it
        for file in trainpath.iterdir():
            image = Image.open(file)
            self.images.append(image)
        
        self.h = height
        self.w = width
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = transforms.ToTensor()(image)
        image = transforms.Resize(size=(self.h, self.w))(image)
        image = 2*image - 1
        return image
        
