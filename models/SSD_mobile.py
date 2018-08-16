import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, xl #from config.py
from .backbones import mobilenetv1, mobilenetv2
from .mobilenetv2 import InvertedResidual
import os

# inherit nn.Module so it have .train()
class SSD_MobN1(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base MobileNetV1 followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: MobileNet v1 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        max_per_image: same as top_k, used in Detection, keep 200 detections per image by default
    """

    def __init__(self, phase, size, base, extras, head, num_classes, cfg, max_per_image):
        super(SSD_MobN1, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        # just create an object above, but need to call forward() to return prior boxes coords
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(256, 20) # L2Norm(512, 20), 256 for mobilenetv1
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])#loc conv layer
        self.conf = nn.ModuleList(head[1])#conf conv layer

        #if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, self.cfg, max_per_image, 0.01, 0.45)

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    For each default box, predict both the shape offsets and the confidences for all object categories
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4] #variance???
        """
        sources = list()# used for storing output of chosen layers, just concat them together
        loc = list()
        conf = list()

        # apply mobilenet v1 to index 5
        for k in range(6):
            x = self.base[k](x)

        s = self.L2Norm(x)#just a kind of normalization
        sources.append(s)

        # apply mobilenet v1 right before avg pool (the last layer in self.features)
        for k in range(6, len(self.base) - 1): # 14
            x = self.base[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # l and c is two conv layers
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # store the output of loc conv layer
            conf.append(c(x).permute(0, 2, 3, 1).contiguous()) # store the output of conf conv layer

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #if self.phase == "test":
        if test:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),   # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# inherit nn.Module so it have .train()
class SSD_MobN2(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base MobileNetV2 followed by the
    added multibox conv layers.  Each multibox layer branches into
    """

    def __init__(self, phase, size, base, extras, head, num_classes, cfg, max_per_image):
        super(SSD_MobN2, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        # just create an object above, but need to call forward() to return prior boxes coords
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(32, 20) # 512
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])#loc conv layer
        self.conf = nn.ModuleList(head[1])#conf conv layer

        #if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, self.cfg, max_per_image, 0.01, 0.45)

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.
        """
        sources = list()# used for storing output of chosen layers, just concat them together
        loc = list()
        conf = list()

        # apply mobilenet v2 to index 6
        for k in range(7):
            x = self.base[k](x)

        s = self.L2Norm(x)#just a kind of normalization
        sources.append(s)

        # apply mobilenet v1 right after avg pool (the last layer in self.features)
        for k in range(7, len(self.base) - 1): # 19
            x = self.base[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # l and c is two conv layers
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # store the output of loc conv layer
            conf.append(c(x).permute(0, 2, 3, 1).contiguous()) # store the output of conf conv layer

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #if self.phase == "test":
        if test:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),   # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to mobileNet for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def mob1_multibox(mob1, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    mob1_source = [5, -2] #two con_dw
    for k, v in enumerate(mob1_source):
        # conv2d (in_channels, out_channels, kernel_size, stride, padding)
        loc_layers += [nn.Conv2d(mob1[v][3].out_channels, #[3] is the last conv within conv_dw
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(mob1[v][3].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):# start k from 2
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return mob1, extra_layers, (loc_layers, conf_layers)

def mob2_multibox(mob2, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    mob2_source = [6, -2] #one InvertedResidual, one conv_1x1_bn
    for k, v in enumerate(mob2_source):
        if isinstance(mob2[v], InvertedResidual):
            out_channels = mob2[v].oup # object InvertedResidual
        else:
            out_channels = mob2[v][0].out_channels # conv_1x1_bn
        # conv2d (in_channels, out_channels, kernel_size, stride, padding)
        loc_layers += [nn.Conv2d(out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):# start k from 2
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return mob2, extra_layers, (loc_layers, conf_layers)

# this is the dict based on which to build the vgg and extras layers one by one
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

def build_mssd(phase, cfg, size=300, num_classes=21, base='m1', max_per_image = 200, width_mult = 1.):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # str(size) will change 300 to '300'
    if base =='m2':
        base_, extras_, head_ = mob2_multibox(mobilenetv2(width_mult),
                                         add_extras(extras[str(size)], int(1280 * width_mult) if width_mult > 1.0 else 1280),
                                         mbox[str(size)], num_classes)
        return SSD_MobN2(phase, size, base_, extras_, head_, num_classes, cfg, max_per_image)
    else:
        base_, extras_, head_ = mob1_multibox(mobilenetv1(),
                                         add_extras(extras[str(size)], 1024),
                                         mbox[str(size)], num_classes)
        return SSD_MobN1(phase, size, base_, extras_, head_, num_classes, cfg, max_per_image)
