import abc
from abc import ABC, abstractmethod, abstractproperty
from multiprocessing.sharedctypes import Value
import constants as constants
import torch
import utils.torch_data_utils as tdu
import utils.common_utils as cu
import utils.torch_utils as tu
import numpy as np

class Data(ABC):
    """This is an abstract class for Dataset
    For us dataset is a tuple (x, y, z, beta, siblings, Z_id, ideal_betas)
    This is a facade for all the data related activities in code.
    """
    def __init__(self, X, y, Z, Beta, Z_ids, *args, **kwargs) -> None:
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.LongTensor(y)
        self.Z = torch.Tensor(Z)

        assert len(Z) == len(y), "If the length of Z and y is not the same, we may have to repeat_interleave somewhere to make the rest of code easy to access."

        self.Beta = torch.LongTensor(Beta)
        self.Z_ids = torch.LongTensor(Z_ids)
        self.num_Z = len(set(self.Z_ids.tolist())) # This is something that will never change in the dataset
        self.classes = list(set(y.tolist()))
        self.num_classes = len(self.classes)
        self.transform = None

        self.unq_beta = self._get_unq_beta()

        self.__init_kwargs(kwargs)

    def __init_kwargs(self, kwargs):
         if constants.TRANSFORM in kwargs.keys():
            self.transform = kwargs[constants.TRANSFORM]

    @property
    def _data_ids(self) -> torch.Tensor:
        return torch.arange(len(self.X))

    @property
    def _Z_ids(self) -> torch.Tensor:
        return self.Z_ids
    @_Z_ids.setter
    def _Z_ids(self, value):
        self.__clear_cache()
        self.Z_ids = value
    

    @property
    def _X(self) -> torch.Tensor:
        return self.X
    @_X.setter
    def _X(self, value):
        self.__clear_cache()
        self.X = value
    
    @property
    def _y(self) -> torch.Tensor:
       return self.y
    @_y.setter
    def _y(self, value):
        self.__clear_cache()
        self.y = value
    
    @property
    def _Z(self) -> torch.Tensor:
        return self.Z
    @_Z.setter
    def _Z(self, value):
        raise ValueError("Have we decided to generate new Objets by means of exploration?")
        self.Z = value
    
    @property
    def _Beta(self) -> torch.Tensor:
        return self.Beta
    @_Beta.setter
    def _Beta(self, value):
        self.__clear_cache()
        self.Beta = value

    @property
    def _num_beta(self):
        return self.Beta.shape[1]

    @property
    def _unq_beta(self):
        return self.unq_beta

    @property
    def _num_data(self):
        return len(self._data_ids)

    @property
    def _num_Z(self):
        return self.num_Z

    @property
    def _Xdim(self):
        return self._X.shape[1]
    
    @property
    def _num_classes(self):
        return self.num_classes
    @property
    def _classes(self):
        return self.classes

    @abstractproperty
    def _list_beta_dims(self):
        """Returns a list of numbers to be used by g_phi to predict the betas.
        """
        raise NotImplementedError()

    def __len__(self):
        return self._num_data
        
# %% some useful functions
    
    def get_instances(self, data_ids:torch.Tensor):
        """Returns ij data ids in order:
            x, y, Beta

        Args:
            data_ids (Tensor): [description]

        Returns:
            X, y, Z, Beta in order
        """
        raise NotImplementedError()
        if not isinstance(data_ids, torch.Tensor):
            raise ValueError("Make everything a Tensor in code")
        return self._X[data_ids], self._y[data_ids], self._Beta[data_ids]
    
    def get_Zgrp_instances(self, zids:torch.Tensor):
        """Finds z id of all the ij instances given in the data_ids
        Then returns all the items in the Z group in order
            x, y, z, Beta

        Args:
            data_ids (np.array): [description]

        Returns:
            X, y, Z, Beta
        """
        raise NotImplementedError()
        if isinstance(zids, int):
            zids = torch.Tensor([zids]).to(torch.int64)
        if not isinstance(zids, torch.Tensor):
            zids = torch.Tensor(zids).to(torch.int64)
        zids = [torch.where(self._Z_ids == entry)[0] for entry in zids]
        zids = torch.stack(zids).flatten()
        return zids, self._X[zids], self._y[zids], self._Beta[zids]

    def get_data_loader(self, batch_size, shuffle, **kwargs):
        """Gets the data loader. We wrap a CustomDataset, get_item of which returns:
        data_id, x, y 
        in order

        Args:
            batch_size (_type_): _description_
            shuffle (_type_): _description_

        Returns:
            dataloader
        """
        custom_ds = tdu.CustomDataset(self._data_ids, self.X, self.y, self.transform)
        custom_loader = tdu.init_loader(custom_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return custom_loader

    @abc.abstractmethod
    def apply_recourse(self, data_id, betas):
        raise NotImplementedError()
    
    @abstractmethod
    def _get_unq_beta(self) -> torch.Tensor:
        assert self._unq_beta is None, "Why are u computing the unq beta again?"
        return torch.LongTensor(cu.cartesian_product([np.arange(entry) for entry in self._list_beta_dims]))

    @staticmethod
    def hash_arm(arm:torch.Tensor):
        return "::".join(map(str, arm.tolist()))
    def hash_ybeta_arm(self, arm:torch.Tensor):
        ybeta = arm[0:self._num_beta+1]
        return "::".join(map(str, ybeta.tolist()))

# %% DataHelper is a wrapper of train, test and val datasets
class DataHelper(ABC):
    def __init__(self, train, test, val, train_test=None, ds_name=None) -> None:
        super().__init__()
        self.train = train
        self.test = test
        self.val = val
        self.train_test = train_test
        self.ds_name = ds_name
    
    @property
    def _train(self) -> Data:
        return self.train
    @_train.setter
    def _train(self, value):
        self.train = value

    @property
    def _train_test(self) -> Data:
        return self.train_test
    @_train_test.setter
    def _train_test(self, value):
        self.train_test = value
    
    @property
    def _test(self) -> Data:
        return self.test
    @_test.setter
    def _test(self, value):
        self.test = value

    @property
    def _val(self) -> Data:
        return self.val
    @_val.setter
    def _val(self, value):
        self.val = value
