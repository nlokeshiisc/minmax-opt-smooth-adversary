import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def init_weights(m:nn.Module):

    def set_params(w):
        if isinstance(w, nn.Linear):
            torch.nn.init.xavier_uniform_(w.weight)
            w.bias.data.fill_(0.01)
    m.apply(set_params)

def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer
    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler
    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")



def _sel_nzro(self, t, sij):
    sel_nonzero = lambda t, sij : torch.squeeze(t[torch.nonzero(sij)])
    res = sel_nonzero(t, sij)
    if res.dim() == t.dim()-1:
        res = torch.unsqueeze(res, 0)        
    return res
    
def _sel_zro(self, t, sij):
    sel_zero = lambda t, sij : torch.squeeze(1-t[torch.nonzero(sij)])
    res = sel_zero(t, sij)
    if res.dim() == t.dim()-1:
        res = torch.unsqueeze(res, 0)        
    return res

def torch_intersect(t1, t2, use_unique=False):
    """Finds the intersection of the two sets

    Args:
        t1 (_type_): _description_
        t2 (_type_): _description_
        use_unique (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    t1 = t1.cuda()
    t2 = t2.cuda()
    t1 = t1.unique()
    t2 = t2.unique()
    t1=set(t1.cpu().numpy())
    t2=set(t2.cpu().numpy())    
    return torch.LongTensor(list(t1.intersection(t2)))

def setminus1d(t1, t2):
    """Computes the Set difference t1 - t2

    Args:
        t1 (_type_): _description_
        t2 (_type_): _description_
    """
    if len(t1) == 0 or len(t2) == 0:
        return t1
    combined = torch.cat((t1, t2, t2)) # t2 should definitely not be returned
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference

def setintersect1d(a, b):
    """Find the intersection of two tensors
    """
    unq_a, unq_b = torch.unique(a), torch.unique(b)
    a_cat_b, counts = torch.cat([unq_a, unq_b]).unique(return_counts=True)
    intersection = a_cat_b[torch.where(counts.gt(1))]
    return intersection



def setdiff1d(t1, t2):
    """Computes the Set difference t1 - t2

    Args:
        t1 (_type_): _description_
        t2 (_type_): _description_
    """
    if len(t1) == 0 or len(t2) == 0:
        return t1
    combined = torch.cat((t1, t2)) # t2 should definitely not be returned
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    # intersection = uniques[counts > 1]
    return difference

def row_equals(tensor_2d: torch.Tensor, row:torch.Tensor):
    """Returns the row ids of tensors that are equal to the row.

    Args:
        tensor_2d (torch.Tensor): _description_
        row (torch.Tensor): _description_
    """
    row = row.squeeze()
    assert tensor_2d.dim() == 2 and row.dim() == 1, "pass a 2d tensor and a row only"
    return torch.where(torch.sum(tensor_2d == row, dim=1) == len(row))[0]