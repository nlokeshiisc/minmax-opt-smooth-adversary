import logging
import constants as constants
import pickle as pkl
from src.data_helper import SyntheticData, SyntheticDataHelper
from src.abstract.abs_data import Data, DataHelper
from src.abstract.abs_cls import ClsHelper
from utils import common_utils as cu
from src.cls import LRClsHelper, ResnetClsHelper
from torch.utils.tensorboard.writer import SummaryWriter

def get_data_helper(dataset_name, logger:logging.Logger = None):
    if logger is not None:
        logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name in [constants.SYN_4DIM_SMALL, constants.SYN_4DIM_XSMALL ,constants.SYN_4DIM_LARGE, constants.SYN_6DIM_SMALL,\
                            constants.SYN_6DIM_LARGE, constants.SYN_4DIM_XLARGE, constants.SYN_6DIM_XLARGE]:
        data_dir = constants.SYN_DIR
        print(f"Loading the dataset from dir: {data_dir}")

        trn_file = f"{dataset_name}_train.pkl"

        if "4dim" in dataset_name:
            tst_file = "syn_data_4dim_test.pkl"
        elif "6dim" in dataset_name:
            tst_file = "syn_data_6dim_test.pkl"

        with open(data_dir / trn_file, "rb") as file:
            train = pkl.load(file)
        with open(data_dir / tst_file, "rb") as file:
            test = pkl.load(file)

        X, Z, Beta, y, Z_ids = train
        train_data = SyntheticData(X=X, y=y, Z = Z, Beta=Beta, Z_ids=Z_ids)

        X, Z, Beta, y, Z_ids = test
        test_data = SyntheticData(X=X, y=y, Z = Z, Beta=Beta, Z_ids=Z_ids)

        dh = SyntheticDataHelper(train_data, test_data, test_data, train_test_data=train_data, 
                                    ds_name=dataset_name)
    
    else:
        raise ValueError("Pass supported datasets only")
    
    if logger is not None:
        logger.info(f"number of reg functions: {dh._train._num_reg_fns}; Domain: {dh._train._list_regfn_dims}")
        logger.info(f"Arms shape: {dh._train._arms.shape}; num_arms: {dh._train._num_arms}; arms dim: {dh._train._arms_dim}")
        logger.info("Printing arm ids for debugging purposes")
        for k, v in dh._train.arm_hash_to_index.items():
            logger.info(f"{k} -- {v}")
        logger.info("Dataset loading completed")


    return dh


def fit_cls(cls_type, cls_name, dh:DataHelper, fit, cls_epochs, logger:logging.Logger=None, *args, **kwargs) -> ClsHelper:

    if cls_type == constants.LOGREG:
        cls_hlpr = LRClsHelper(dh = dh, **kwargs)
    elif cls_type == constants.RESNET:
        cls_hlpr = ResnetClsHelper(dh=dh, **kwargs)
    else:
        raise NotImplementedError()

    if logger is not None:
        logger.info(f"cls is: {cls_type}")

    # fit
    if fit == True:
        print("Fitting cls")
        if logger is not None:
            logger.info(f"Fittig nntheta for {cls_epochs} Epochs")

        cls_hlpr.fit_data(epochs=cls_epochs, logger=logger, **kwargs)

        test_acc = cls_hlpr.accuracy()
        print(f"Test Accuracy after fitting cls: {test_acc}")
        if logger is not None:
            logger.info(f"Test Accuracy after fitting cls: {test_acc}")

        cls_hlpr.save_model_defname(suffix=cls_name)

    # load
    else:
        if logger is not None:
            logger.info("Loading nth model as fit = False")
        
        if cls_name is not None: # Donot load the model is we dont pass a models_defname
            cls_hlpr.load_model_defname(suffix=cls_name, logger=logger)

    print("nnth helper Ready!")
    if logger is not None:
        logger.info("nnth Ready")
    return cls_hlpr
