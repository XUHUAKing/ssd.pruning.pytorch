'''
    This file is implementation for mobileNet v2
'''

import torch.nn as nn
import math

## kernel_size=(3, 3) pad = 1
def conv_bn(inp, oup, stride):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

## kernel_size=(1, 1) stride = 1 pad = 0 increase / reduce dims
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup  # using residual connect when input_channel = output_channel
        self.oup = oup # for SSD multibox

        self.conv = nn.Sequential(
                # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
                # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
                # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
                    return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.): # width_mult also known as depth multiplier
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s # t: expand_ratio, c:, n: how many blocks of this type in total, s: stride
            [1, 16, 1, 1],
            [6, 24, 2, 2], # 2nd halfing
            [6, 32, 3, 2], # 3rd halfing
            [6, 64, 4, 2], # 4th halfing
            [6, 96, 3, 1],
            [6, 160, 3, 1], # 5th halfing is disabled for detection
            # [6, 160, 3, 2], # 5th halfing
            [6, 320, 1, 1],
        ] # 1 + 2 + 3 + 4 + 3 + 3 + 1 = 17 blocks in total

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # the first block is different: uses a regular 3×3 convolution with 32 channels instead of the expansion layer
        self.features = [conv_bn(3, input_channel, 2)] # 1st halfing
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel)) # after this layer, connect to extra layers in SSD
        self.features.append(nn.AvgPool2d(int(input_size/32))) # don't used in backbone
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier, this won't be used in backbone
        self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(self.last_channel, n_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def mobilev2_layers(self):
        # return a list containing conv_bn or InvertedResidual object one by one
        layers = []
        layers += self.features.children()
        return layers
