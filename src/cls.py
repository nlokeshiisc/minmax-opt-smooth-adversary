import torch
from src.abstract import abs_cls
from src.abstract import abs_data
from src.models import LRModel
import utils.torch_utils as tu
import utils.common_utils as cu
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from tqdm import tqdm

CLS_STEP = 0

class LRClsHelper(abs_cls.ClsHelper):  
    def __init__(self, dh:abs_data.DataHelper, *args, **kwargs) -> None:
        self.in_dim = dh._train._Xdim
        self.n_classes = dh._train._num_classes
        model = LRModel(in_dim=self.in_dim, n_classes=self.n_classes, *args, **kwargs)
        super(LRClsHelper, self).__init__(model, dh, *args, **kwargs)

        self.init_model()
    
    def init_model(self):
        tu.init_weights(self._model)
        self._model.to(cu.get_device())
        self._optimizer = optim.AdamW([
            {'params': self._model.parameters()},
        ], lr=self._lr)

    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            trn_wts ([type], optional): [description]. Defaults to None. weights to be associated with each traning sample.
            epochs ([type], optional): [description]. Defaults to None. 
            steps ([type], optional): [description]. Defaults to None.
        """
        assert not(epochs is not None and steps is not None), "We will run either the specified SGD steps or specified epochs over data. We cannot run both"
        assert not(epochs is None and steps is None), "We need atleast one of steps or epochs to be specified"

        global CLS_STEP
        global_step = 0
        total_sgd_steps = np.inf
        total_epochs = 10
        if steps is not None:
            total_sgd_steps = steps
        if epochs is not None:
            total_epochs = epochs

        self._model.train()

        if loader is None:
            loader = self._trn_loader
        
        if trn_wts is None:
            trn_wts = torch.ones(len(loader.dataset)).to(cu.get_device())
        assert len(trn_wts) == len(loader.dataset), "Pass all weights. If you intend not to train on an example, then pass the weight as 0"
    
        for epoch in range(total_epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (batch_ids, x, y) in enumerate(loader):
                global_step += 1
                if global_step == total_sgd_steps:
                    if logger is not None:
                        logger.info("Aborting fit as we matched the number of SGD steps required!")
                    return

                x, y, batch_ids = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), batch_ids.to(cu.get_device(), dtype=torch.int64)
                self._optimizer.zero_grad()
                
                cls_out = self._model.forward(x)
                loss = self._xecri_perex(cls_out, y)
                loss = torch.dot(loss, trn_wts[batch_ids]) / torch.sum(trn_wts[batch_ids])
                
                loss.backward()
                self._optimizer.step()

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)
            
            epoch_acc = self.accuracy()
            print(f"Epoch: {epoch} accuracy: {epoch_acc}")
            if self._sw is not None:
                self._sw.add_scalar("nnth_Epoch_Acc", epoch_acc, CLS_STEP)
            if logger is not None:
                logger.info(f"Epoch: {CLS_STEP} accuracy: {epoch_acc}")
        
            CLS_STEP += 1

    @property
    def _def_name(self):
        return "logreg"


class ResnetClsHelper(abs_cls.ClsHelper):  
    def __init__(self, dh:abs_data.DataHelper, *args, **kwargs) -> None:
        self.n_classes = dh._train._num_classes
        model = LRModel(in_dim=self.in_dim, n_classes=self.n_classes, *args, **kwargs)
        super(LRClsHelper, self).__init__(model, dh, *args, **kwargs)

        self.init_model()
    
    def init_model(self):
        tu.init_weights(self._model)
        self._model.to(cu.get_device())
        self._optimizer = optim.AdamW([
            {'params': self._model.parameters()},
        ], lr=self._lr)

    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            trn_wts ([type], optional): [description]. Defaults to None. weights to be associated with each traning sample.
            epochs ([type], optional): [description]. Defaults to None. 
            steps ([type], optional): [description]. Defaults to None.
        """
        assert not(epochs is not None and steps is not None), "We will run either the specified SGD steps or specified epochs over data. We cannot run both"
        assert not(epochs is None and steps is None), "We need atleast one of steps or epochs to be specified"

        global CLS_STEP
        global_step = 0
        total_sgd_steps = np.inf
        total_epochs = 10
        if steps is not None:
            total_sgd_steps = steps
        if epochs is not None:
            total_epochs = epochs

        self._model.train()

        if loader is None:
            loader = self._trn_loader
        
        if trn_wts is None:
            trn_wts = torch.ones(len(loader.dataset)).to(cu.get_device())
        assert len(trn_wts) == len(loader.dataset), "Pass all weights. If you intend not to train on an example, then pass the weight as 0"
    
        for epoch in range(total_epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (batch_ids, x, y) in enumerate(loader):
                global_step += 1
                if global_step == total_sgd_steps:
                    if logger is not None:
                        logger.info("Aborting fit as we matched the number of SGD steps required!")
                    return

                x, y, batch_ids = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), batch_ids.to(cu.get_device(), dtype=torch.int64)
                self._optimizer.zero_grad()
                
                cls_out = self._model.forward(x)
                loss = self._xecri_perex(cls_out, y)
                loss = torch.dot(loss, trn_wts[batch_ids]) / torch.sum(trn_wts[batch_ids])
                
                loss.backward()
                self._optimizer.step()

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)
            
            epoch_acc = self.accuracy()
            print(f"Epoch: {epoch} accuracy: {epoch_acc}")
            if self._sw is not None:
                self._sw.add_scalar("nnth_Epoch_Acc", epoch_acc, CLS_STEP)
            if logger is not None:
                logger.info(f"Epoch: {CLS_STEP} accuracy: {epoch_acc}")
        
            CLS_STEP += 1
    
    @property
    def _def_name(self):
        return "resnet"

