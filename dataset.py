import os
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class DenoisingDiffusionCelebHQ(Dataset):

    def __init__(self, root:str, transform:transforms=None):
        self.files = []
        for root_path, dirs, files in os.walk(root):
            for file in files:
                self.files.append(os.path.join(root_path, file))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        img_path = self.files[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img
    

def get_data_loaders(root:str, img_size:int=512, batch_size:int=8, num_workers:int=12, prefetch_factor:int=2):

    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'val')

    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    train_data = DenoisingDiffusionCelebHQ(root=train_dir, transform=transform)
    test_data = DenoisingDiffusionCelebHQ(root=test_dir, transform=transform)

    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,  
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,  
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    return train_loader, test_loader