#!/usr/bin/python3

##### Note #####
# If using kaggle APIs for the first time, you should follow these steps for installation and Authentication
# 1.pip install kaggle
# 2.cd ~/.kaggle
# 3.homepage www.kaggle.com -> Your Account -> Create New API token
# 4.mv ~/Downloads/kaggle.json ./
# 5.chmod 600 ./kaggle.json

import kaggle
import os
import cv2
import numpy as np 
from zipfile import ZipFile

# get_data function return train_images, train_labels, test_images, test_labels
# train_images has size (300,300,3)
# parameters:
#           dataset_name --> can be either 'dogs' --> download https://www.kaggle.com/jessicali9530/stanford-dogs-dataset
#                                          'cats' --> download https://www.kaggle.com/ma7555/cat-breeds-dataset
#           number_of_breeds --> cats dataset has 67 different cat breeds and dogs dataset has 120 dogs breeds 
#                                so this parameter can be used to get only subset of the data
#           download --> set true if the data was not downloaded before 

def get_data(dataset_name='dogs', number_of_breeds=5, download=True):
    # Assert if dataset_name is not 'dogs' or 'cats'
    assert dataset_name == 'cats' or dataset_name == 'dogs', "dataset_name should either be \'dogs\' or \'cats\'"

    if dataset_name == 'dogs':
        kaggle_name = 'jessicali9530/stanford-dogs-dataset'
        destination_dir = "dogs"
        zip_filename = 'stanford-dogs-dataset.zip'
        skip_index = 2

    elif dataset_name == 'cats':
        kaggle_name = 'ma7555/cat-breeds-dataset'
        destination_dir = "cats"
        zip_filename = 'cat-breeds-dataset.zip'
        skip_index = 1

    if download:
        kaggle.api.authenticate()
        # Checks first if the file exists, if not downloads it.
        kaggle.api.dataset_download_files(kaggle_name, path='.', unzip=False, quiet=False)

        with ZipFile(zip_filename, 'r') as zipObj:
            zipObj.extractall(destination_dir)

    images = []
    labels = []

    for i, (subdir, _, files) in enumerate(os.walk(destination_dir + os.sep + "images/")):
        # skip subdirectories as they are root and not images subdirs
        if i < skip_index:
            continue

        if i-skip_index == number_of_breeds:
            break

        for f in files:
            filepath = subdir + os.sep + f
            if filepath.endswith(".jpg"):
                img = cv2.imread(filepath)
                img = cv2.resize(img, (300,300))

                # divide by 255.0 to convert values between 0 and 1
                img = img/255.0

                images.append(img)
                labels.append(i-skip_index) # subtract from skip_index to start labeling from 0

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)

    # Shuffle the data
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # Divide into 80% training and 20% testing data
    train_index = int(0.8*images.shape[0])

    train_images  = images[0:train_index]
    train_labels = labels[0:train_index]
    test_images = images[train_index:]
    test_labels = labels[train_index:]

    return train_images, train_labels, test_images, test_labels