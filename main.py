import utils.common_utils as cu
import logging
import constants as constants
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
import src.main_helper as main_hlpr

sw = SummaryWriter(constants.TB_DIR / "fonts_expt")

logging.basicConfig(filename= str((constants.LOG_DIR / f"fonts_expt.log").absolute()),
                        format='%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s',
                        filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Datahelper
dh = main_hlpr.get_data_helper(dataset_name=constants.FONTSDS, logger=logger)

# Classifier
cls_kwargs = {
    constants.SW: sw,
    constants.BATCH_SIZE: 32,
    constants.LRN_RATE: 1e-3,
    constants.SCHEDULER: True,
    constants.MOMENTUM: 0.9,
}
cls_model_type = constants.CNN
cls_model_name = "fonts_net"
fit_cls = True
cls_hlpr = main_hlpr.fit_cls(cls_type=cls_model_type, cls_name=cls_model_name,
                                dh=dh, cls_epochs=30,
                                fit=fit_cls, **cls_kwargs)
