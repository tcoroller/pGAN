"""
This file contains the batcher classes F2305Batcher, A2209Batcher and SynthBatcher.
F2305Batcher and A2209Batcher are the same class but have different names for the sake of clarity.
They take the preprocessed dataset from restricted_preprocessing as input.
Can optionnaly take transforms as input for data augmentation (on applied only on the image)
and casting the image to other data type (like casting to tensor with torchvision.transforms.ToTensor())


The batcher loads the image for a given path and returns a dictionnary with
    - dict['image'] as the path to the saved image (.npy file)
    - dict['im'] as the image as a numpy array (or as a tensor if transform ToTensor is applied)
    - dict['region'] as the region label
"""

import os
import numpy as np
from torch.utils.data import Dataset

class F2305Batcher(Dataset):
    """
    F2305 Batcher
    """
    def __init__(self, dataset, scan_path, transform=None):
        '''initialize with dataset, path to data and optionally tranforms'''
        self.dataset = dataset
        self.scan_path = scan_path
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # read record from the dictionnary
        record = self.dataset[idx]
        # load image (stored as .npy) from path in dictionnary
        record['im'] = np.load(os.path.join(self.scan_path, record['image']))
        # apply transform to the image if specified
        if self.transform is not None:
            record['im'] = self.transform(record['im'])
        # returns record with: image path at record['image']; image content at record['im'];
        # image label as record['region']
        return record


class A2209Batcher(Dataset):
    """
    A2209 Batcher
    """
    def __init__(self, dataset, scan_path, transform=None):
        '''initialize with dataset, path to data and optionally tranforms'''
        self.dataset = dataset
        self.scan_path = scan_path
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # read record from the dictionnary
        record = self.dataset[idx]
        # load image (stored as .npy) from path in dictionnary
        record['im'] = np.load(os.path.join(self.scan_path, record['image']))
        # apply transform to the image if specified
        if self.transform is not None:
            record['im'] = self.transform(record['im'])
        # returns record with: image path at record['image']; image content at record['im'];
        # image label as record['region']
        return record


class SynthBatcher(Dataset):
    """
    Synthetic Batcher
    """
    def __init__(self, dataset, scan_path, transform=None):
        '''initialize with dataset, path to data and optionally tranforms'''
        self.dataset = dataset
        self.scan_path = scan_path
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # create dictionnary
        record = {}
        # image path as record['image']
        record["image"] = self.dataset[idx]
        # parse label from file_name
        record["region"] = int(record["image"].replace(".npy", "")[-1])
        # load image from .npy file
        record['im'] = np.load(os.path.join(self.scan_path, record['image'])).squeeze(0)
        # Apply transform if specified
        if self.transform is not None:
            record['im'] = self.transform(record['im'])
        # return full dict
        return record
