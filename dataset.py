import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# similar to the pytorch image folder dataset loader
class ImageFolder(Dataset):
    """The class is used to load images 
    """
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        image = Image.open(path).convert('RGB')
        image = self.transforms(image)

        return image, label


# TODO: what is the input size
# code similar to https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def transform_image(mode='train', input_size=300):

    # Normalize a tensor image with mean and standard deviation. 
    # Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, 
    # this transform will normalize each channel of the input torch.*Tensor
    #  i.e. output[channel] = (input[channel] - mean[channel]) / std[channel]

    if mode == 'train':
        return transforms.Compose([
               transforms.RandomResizedCrop(input_size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return  transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


def load_images(data_dir, classes):
    """Function loads images from the classes
    """
    all_data = []
    for i, _class in enumerate(classes):
        _path = os.path.join(data_dir, _class)
        images = os.listdir(_path)
        for image in images:
            all_data.append((os.path.join(_path, image), i))
    
    return all_data