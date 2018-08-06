import torch
import torch.nn as nn
from .resnet import *
from .mobilenetv1 import *
from .mobilenetv2 import *


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# cfg is a list of string
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # pool will not be affetced by pruning because no need for in_channels
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)# fc6 and fc7 are changed to conv layers
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}


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
