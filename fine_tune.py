import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from dataset import ImageFolder, transform_image, load_images
from utils import train, get_dataloaders, save_model, create_dirs

from train import TRAIN_CLASSES as DOG_CLASSES

TRAIN_CLASSES = [
    'Abyssinian',
    'Bengal',
    'Tabby'    
    ]

DATASET_DIR = 'cats/images'


def main(model_name):
    print(f'Fine-tuning {model_name} model for cats')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load the resnet50 model
    print('loading the pretrained model')
    model = models.resnet18(pretrained=True)

    num_feat = model.fc.in_features
    if model_name == 'dogs':
        model.fc = nn.Linear(num_feat, len(DOG_CLASSES))
        model.load_state_dict(torch.load('save/dogs.pt'))

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(num_feat, len(TRAIN_CLASSES))

    # load the dataset
    print('loading cats dataset')
    dataloaders_dict = get_dataloaders(DATASET_DIR, TRAIN_CLASSES)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-03)
    print('fine-tuning the model')
    model, val_loss = train(model, dataloaders_dict, loss, optimizer, device, no_of_epochs=20)

    if model_name == 'dogs':
        model_path = 'save/dogs_to_cats.pt'
    else:
        model_path = 'save/imagenet_to_cats.pt'
    
    save_model(model, model_path)

    print('done!')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dogs or imagenet
    parser.add_argument('--pretrained-model', default='dogs')
    args = parser.parse_args()
    
    # fine-tune model
    main(args.pretrained_model)
