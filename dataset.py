from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms

class CelebDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        folderpath = Path.cwd()/datapath
        self.images = []
        for i, file in enumerate(folderpath.iterdir()):
            image = Image.open(file)
            self.images.append(image)
            if i == 2:
                break
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = transforms.ToTensor()(image)
        image = transforms.Resize(size=(216, 176))(image)
        image = 2*image - 1
        return image
        
