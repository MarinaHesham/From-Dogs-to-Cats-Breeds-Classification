
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse

from dataset import ImageFolder, transform_image, load_images
from utils import train, get_dataloaders, save_model, create_dirs

TRAIN_CLASSES = {"dogs": [
    'n02085620-Chihuahua',
    'n02088364-beagle',
    'n02094258-Norwich_terrier',
    'n02092002-Scottish_deerhound'
    ], "cats": [
    'Abyssinian',
    'Bengal',
    'Tabby'
    #'Tonkinese'
    #'Tortoiseshell'   
    ]}

DATASET_DIR = {"dogs": 'dogs/images/Images', "cats": 'cats/images'}


def main(dataset_name):
    print('Training resnet model for', dataset_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the dataset
    print('loading', dataset_name, 'dataset')
    dataloaders_dict = get_dataloaders(DATASET_DIR[dataset_name], TRAIN_CLASSES[dataset_name])
    
    # load the resnet18 model
    print('loading the resnet model')
    model = models.resnet18()
    num_feat = model.fc.in_features
    print(len(TRAIN_CLASSES[dataset_name]))
    model.fc = nn.Linear(num_feat, len(TRAIN_CLASSES[dataset_name]))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-04)
    print('training the model')
    model, val_loss = train(model, dataloaders_dict, loss, optimizer, device, no_of_epochs=20)

    model_path = 'save/dogs.pt'
    create_dirs(model_path)
    print('saving the model')
    save_model(model, model_path)
    
    print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dogs or cats
    parser.add_argument('--dataset_name', default='dogs')
    args = parser.parse_args()
    
    # fine-tune model
    main(args.dataset_name)
