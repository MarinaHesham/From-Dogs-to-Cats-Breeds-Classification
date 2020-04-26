
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from dataset import ImageFolder, transform_image, load_images
from utils import train, get_dataloaders, save_model, create_dirs

TRAIN_CLASSES = [
    'n02085620-Chihuahua',
    'n02088364-beagle'    
    ]

DATASET_DIR = 'dogs/images/Images'


def main():
    print('Training resnet model for dogs')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the dataset
    print('loading dogs dataset')
    dataloaders_dict = get_dataloaders(DATASET_DIR, TRAIN_CLASSES)
    
    # load the resnet50 model
    print('loading the resnet model')
    model = models.resnet50()
    num_feat = model.fc.in_features
    model.fc = nn.Linear(num_feat, len(TRAIN_CLASSES))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-03)
    print('training the model')
    model, val_loss = train(model, dataloaders_dict, loss, optimizer, device, no_of_epochs=20)

    model_path = 'save/dogs.pt'
    create_dirs(model_path)
    print('saving the model')
    save_model(model, model_path)
    
    print('done!')


if __name__ == "__main__":
    main()