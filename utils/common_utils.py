import random
import torch
import numpy as np
from copy import deepcopy
import json
import constants

def set_seed(seed:int):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return "cuda:0"
    # return "cpu"

def set_cuda_device(gpu_num: int):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

def insert_kwargs(kwargs:dict, new_args:dict):
    assert type(new_args) == type(kwargs), "Please pass two dictionaries"
    merged_args = kwargs.copy()
    merged_args.update(new_args)
    return merged_args

def cartesian_product(arrays:list):
    la = len(arrays)
    dtype = int
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def dict_print(d:dict):
    d_new = deepcopy(d)

    def cast_str(d_new:dict):
        for k, v in d_new.items():
            if isinstance(v, dict):
                d_new[k] = cast_str(v)
            d_new[k] = str(v)
        return d_new
    d_new = cast_str(d_new)

    str_d = {}
    for k, v in d_new.items():
        str_d[str(k)] = v

    pretty_str = json.dumps(str_d, sort_keys=False, indent=4)
    print(pretty_str)
    return pretty_str

def sort_binarr_allrows(b):
    b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
    return b[np.argsort(b_view.ravel())]

def manipulate_config(config, args, unknown_args):
    #overriding config.
    for override_config in unknown_args:
        parts = override_config.split(":")
        key = parts[0]
        value = parts[1]
        
        if "." in key:
            key_parts = key.split(".")
            primary_key = key_parts[0]
            secondary_key = key_parts[1]
            try:
                config[primary_key][secondary_key] = eval(value)
            except:
                config[primary_key][secondary_key] = value
        else:
            config[key] = value
    print(config, flush=True)
    return config
