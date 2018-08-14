'''
    This file is implementation for vgg
'''
import torch
import torch.nn as nn

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# cfg is a list of string
def vgg(cfg, i=3, refine = False, batch_norm=False):
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
    if not refine:
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    else:
        pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)# fc6 and fc7 are changed to conv layers
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

class VGG(nn.Module):
    def __init__(self, refine = False, i=3, batch_norm=False):
        super(VGG, self).__init__()
        self.features = vgg(vgg_base['300'], i, refine = refine, batch_norm = batch_norm)
        # no classifier for this self-designed VGG

    def forward(self, x):
        return x

    def vgg_layers(self):
        layers = []
        layers += self.features.children()
        return layers # has already exclude the last layer
