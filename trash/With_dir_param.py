import os
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cnw.utilities.datasets import Dataset

# Create data loaders for train and test sets
train_dataset = CropAndWeedDataset(images_dir='../data/images', bboxes_dir='../data/bboxes/CropAndWeed/CropAndWeed/CropOrWeed2', filenames=train_filenames)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CropAndWeedDataset(images_dir='../data/images', bboxes_dir='../data/bboxes/CropOrWeed2', filenames=test_filenames)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


for data in train_dataloader:
    images, bounding_boxes = data['image'], data['bounding_boxes']

    # Print the image
    plt.imshow(images[0])
    plt.show()

    # Print the bounding boxes
    print(bounding_boxes[0])
    break