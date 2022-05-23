import torch.utils.data as data_utils
import constants
import torch
from PIL import Image
import os
import sys

class CustomDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data_ids, X, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.y = y
        self.transform = transform


    def __getitem__(self, index):
        data_id, x, y= self.data_ids[index], self.X[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, y

    def __len__(self):
        return len(self.y)

def get_loader_subset(loader:data_utils.DataLoader, subset_idxs:list, batch_size=None, shuffle=False):
    """Returns a data loader with the mentioned subset indices
    """
    subset_ds = data_utils.Subset(dataset=loader.dataset, indices=subset_idxs)
    if batch_size is None:
        batch_size = loader.batch_size
    return data_utils.DataLoader(subset_ds, batch_size=batch_size, shuffle=shuffle)

def init_loader(ds:data_utils.Dataset, batch_size, shuffle=False, **kwargs):
    if constants.SAMPLER in kwargs:
        return data_utils.DataLoader(ds, batch_size=batch_size, sampler=kwargs[constants.SAMPLER])
    else:
        return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class DatasetNumpy(torch.utils.data.Dataset):
    """This is Dataset taken from Vihari's code
    """
    def __init__(self, np_x, np_y, transform=None):
        self.np_x = np_x
        self.np_y = np_y
        self.transform = transform
        
    def __getitem__(self, index):
        x = Image.fromarray(self.np_x[index]).convert("RGB")
        y = self.np_y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.np_x)
