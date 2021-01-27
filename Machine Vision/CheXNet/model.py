"""
CheXNet Model Implementation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataset
from sklearn.metrics import roc_auc_score


CKPH_PATH = 'model.ckt'
NUM_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
               'Mass', 'Nodule', 'Pneumonia''Pneumothorax', 'Consolidation',
               'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening''Hernia']

DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 64


def main():

    cudnn.benchmark = True

    # initialize and load the model


class DenseNet121(nn.Module):
    """
    Model Modified

    The architecture of our model is the same as standard DenseNet121
    except the clasifier layer which has an  additional sigmoid function
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()
