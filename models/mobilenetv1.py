'''
    This file is implementation for mobileNet v1
'''

import torch
import torch.nn as nn

## kernel_size=(3, 3) pad = 1
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    # all conv layers followed by RELU6
    return nn.Sequential(
        # 3x3 depthwise convolution
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # 1x1 pointwise convolution
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), # index = 3
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # depthwise layer with stride 2: reduce width and height of data
        # pointwide layer: sometimes double the num of channels in the data
        self.features = nn.Sequential(
            conv_bn(  3,  32, 2), # regular convolutional layer with stride 2 - 1st halfing
            conv_dw( 32,  64, 1), # depthwise layer + pointwise layer that doubles the number of channels
            conv_dw( 64, 128, 2), # depthwise layer with stride 2 + pointwise layer that doubles the number of channels - 2nd halfing
            conv_dw(128, 128, 1), # depthwise layer + pointwise layer
            conv_dw(128, 256, 2), # depthwise layer with stride 2 + pointwise layer that doubles the number of channels - 3rd halfing
            conv_dw(256, 256, 1), # source for detection here -- index = 5
            conv_dw(256, 512, 2), # 4th halfing
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1), # 5th halfing changed to stride = 1 for detection task
#             conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7), # source for detection here -- index = 14
        )
        # building classifier (i.e. fc)
        self.classifier = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

    def mobilev1_layers(self):
        # return a list containing conv_bn or conv_dw object one by one
        return self.features.children()
