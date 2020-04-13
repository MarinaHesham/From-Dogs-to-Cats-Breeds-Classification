#!/usr/bin/python3

from preprocessing import get_data
from model import run_model


# Get Dogs Dataset
print("Processing Dogs dataset")
train_images, train_labels, test_images, test_labels = get_data(dataset_name='dogs', download=True, number_of_breeds=5)
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

# Get Cats Dataset
print("Processing Cats dataset")
train_images, train_labels, test_images, test_labels = get_data(dataset_name='cats', download=True, number_of_breeds=5)
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

# Run the model
print("Running Model")
run_model(train_images, train_labels, test_images, test_labels)
