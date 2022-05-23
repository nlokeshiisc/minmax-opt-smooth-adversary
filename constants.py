from pathlib import Path
from torchvision import transforms

TOL = 1e-6

DATASET = "dataset"
MODEL_NAME = "model_name"
MODEL_TYPE = "model_type"
FONTSDS = "fonts_dataset"

LRN_RATE = "lr"
MOMENTUM = "momentum"
OPTIMIZER = "optimizer"
BATCH_NORM = "batch_norm"
SAMPLER = "sampler"
SCRATCH = "scratch"
NNARCH = "nn_arch"

TRANSFORM = "transform"
SCHEDULER = "scheduler"
SCHEDULER_TYPE = "scheduler_type"
EPOCHS = "epochs"

SW = "summarywriter"
BATCH_SIZE = "batch_size"

EXPT_NAME = "experiment_name"

LOGREG = "logistic_regression"
FNN = "fully_conncted_neural_net"
CNN = "convolutional_neural_net"
RESNET = "resnet"

CLS_SPECS = "classifier_specs"
GENERAL_SPECS = "general_specs"

CLS = "classifier"

FONTS_DIR = Path("dataset/fonts")

EXPT_NAME = "expt_name"
SUFFIX = "suffix_name"

RESNET_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

TB_DIR = Path("gp_src/tblogs/")
LOG_DIR = Path("gp_src/results/logs/")
LOG_FILE = "logger_file"
GPUID = "gpu_id"
SEED = "random_seed"