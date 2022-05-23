import torch.nn as nn
import torch
import utils.common_utils as cu
import constants as constants
from torchvision import models as tv_models

class LRModel(nn.Module):
    def __init__(self, in_dim, n_classes, *args, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes

        self.__init_model()

        self.sm = nn.Softmax(dim=1)

    def __init_model(self):
        self.in_layer = nn.Linear(self.in_dim, out_features=self.n_classes)


    def forward_proba(self, input):
        out = self.in_layer(input)
        return self.sm(out)
    
    def forward(self, input):
        return self.in_layer(input)
    
    def forward_labels(self, input):
        probs = self.forward_proba(input)
        probs, labels = torch.max(probs, dim=1)
        return labels

class ResNET(nn.Module):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        self.init_model()
        self.sm = nn.Softmax(dim=1)

    def init_model(self):
        self.resnet_features =  tv_models.resnet18(pretrained=False)
        print("Loading resnet cls with pretrain = False")
        self.emb_dim = self.resnet_features.fc.in_features
        self.resnet_features.fc = nn.Linear(self.emb_dim, self.out_dim)

    def forward_proba(self, input):
        out = self.resnet_features(input)
        return self.sm(out)
    
    def forward(self, input):
        return self.resnet_features(input)
    
    def forward_labels(self, input):
        probs = self.forward_proba(input)
        probs, labels = torch.max(probs, dim=1)
        return labels