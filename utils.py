import copy
import os
import torch
import torch.nn as nn
from dataset import load_images, ImageFolder, transform_image


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def get_dataloaders(dir_path, train_classes, batch_size=128):
    image_label = load_images(dir_path, train_classes)

    # split the dataset into train, validation and test
    num_images = len(image_label)
    train_split, val_split = int(num_images * 0.8), int(num_images * 0.9)
    # TODO: shuffle the images
    train, val, test = image_label[:train_split], image_label[train_split:val_split], image_label[val_split:]
    
    # make them into pytorch datasets
    train_dataset = ImageFolder(train, transform_image('train'))
    valid_dataset = ImageFolder(val, transform_image('valid'))
    test_dataset = ImageFolder(test, transform_image('test'))

    dataloaders_dict = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1),
        "val": torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    }

    return dataloaders_dict


def create_dirs(model_path):
    dir_path = os.path.dirname(model_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def train(model, dataloaders_dict, loss_fn, optimizer, device, no_of_epochs=25):
    model.to(device)
    val_loss = []
    best_model = None
    for epoch in range(no_of_epochs):
        print(f'Epochs: {epoch + 1}')
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # 
            total_loss = 0.
            correct = 0
            for batch in dataloaders_dict[phase]:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
                total_loss += loss.item()
                correct += torch.sum(preds == labels.data)
            
            epoch_loss = total_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = correct.double() / len(dataloaders_dict[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # check if the validation loss is the best 
            if phase == 'val':
                if not val_loss:
                    # save model 
                    print("save model")
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    min_loss = min(val_loss)
                    if epoch_loss < min_loss:
                        best_model = copy.deepcopy(model.state_dict())

                val_loss.append(epoch_loss)
        
    model.load_state_dict(best_model)

    return model, val_loss
