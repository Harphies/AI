"""
Read images and corresponding labels.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory/root folder
            image_list_files: path to the file containing images
            with corresponding labels
            transform: optional transform to be applied on a sample
        """
        image_names = []
        labels = []
        with open(image_list_file, 'r') as all_images:
            for image in all_images:
                image_parts = image.split()
                image_name = image_parts[0]
                label = image_parts[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of image

        Returns:
        image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor.float(label)

    def __len__(self):
        return len(self.image_names)
