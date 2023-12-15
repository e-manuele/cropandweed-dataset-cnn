
import os

import torch
import numpy as np
from torch.utils.data import  Dataset

from PIL import Image



class CropAndWeedDataset(Dataset):

    def __init__(self, data_root):
        super().__init__()

        self.image_paths = []
        self.label_paths = []

        for image_path in os.listdir(os.path.join(data_root, "../images")):
            self.image_paths.append(os.path.join(data_root, "../images", image_path))

        for label_path in os.listdir(os.path.join(data_root, "labelIds")):
            self.label_paths.append(os.path.join(data_root, "labelIds", label_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = np.asarray(image)
        image = image.copy()
        image = torch.from_numpy(image)



        label = Image.open(self.label_paths[index])
        label = np.asarray(label)
        label = label.copy()
        label = torch.from_numpy(label)

        return image, label


dataset = CropAndWeedDataset("data")
print(len(dataset))
image, label = dataset[0]
