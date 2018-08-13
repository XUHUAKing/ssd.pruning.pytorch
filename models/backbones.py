import torch
import torch.nn as nn
from .vgg import *
from .resnet import *
from .mobilenetv1 import *
from .mobilenetv2 import *


def vgg():
    model = VGG()
    model.load_state_dict(torch.load('weights/vgg16_reducedfc.pth'))
    return model.vgg_layers() # exclude fc already

# This function is derived from torchvision ResNet
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# cfg is a list of string
def resnet():
    model = resnet50()
    model.load_state_dict(torch.load('weights/resnet50-19c8e357.pth')) # load pretrained model for detection
    return model.resnet_layers() # exclude fc but keep weights

# return a list of model features for mobileNet v1
def mobilenetv1():
    model = MobileNetV1()
    model.load_state_dict(torch.load('weights/mobilev1_withfc.pth')) # load pretrained model for detection
    return model.mobilev1_layers() # exclude fc but keep weights

# return a list of model features for mobileNet v2
def mobilenetv2(width_mult = 1.):
    model = MobileNetV2(width_mult = width_mult) # n_class and input_size is not used in backbone
    model.load_state_dict(torch.load('weights/mobilev2_withfc.pth')) # load pretrained model for detection, only pretrained for width_multi = 1
    return model.mobilev2_layers() # exclude fc but keep weights
