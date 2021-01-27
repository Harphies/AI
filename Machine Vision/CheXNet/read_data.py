"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os


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
