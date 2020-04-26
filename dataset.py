import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ImageFolder(Dataset):
    """The class is used to load images 
    """
    def __init__(self, path, classes, transforms, stage='train'):
        self.data = []
        self.dir_path = os.path.join(path, stage)
        for i, c in enumerate(classes):
            cls_path = os.path.join(self.dir_path, c)
            images = os.listdir(cls_path)
            for image in images:
                self.data.append((os.path.join(cls_path, image), i))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        image = Image.open(path).convert('RGB')
        image = self.transforms(image)

        return image, label
