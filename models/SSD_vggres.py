import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, xl #from config.py
from .backbones import vgg, vgg_base, resnet
import os

# inherit nn.Module so it have .train()
class SSD_VGG(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_VGG, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if num_classes == 25: # for VOC_xlab_products dataset
            self.cfg = xl
        else:
            self.cfg = (coco, voc)[num_classes == 21]#when num_classes==21, i.e. true/[1], then voc is chosen
        self.priorbox = PriorBox(self.cfg)
        # just create an object above, but need to call forward() to return prior boxes coords
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])#loc conv layer
        self.conf = nn.ModuleList(head[1])#conf conv layer

        #if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        # top_k = (300, 200)[args.dataset == 'COCO'], 200 here is top_k
        self.detect = Detect(num_classes, 0, self.cfg, 200, 0.01, 0.45)

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

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)# use vgg[k] as a function because it is a layer

        s = self.L2Norm(x)#just a kind of normalization
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
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
class SSD_RESNET(nn.Module):
    """
    Single Shot Multibox Architecture
    The network is composed of a base RESNET50 network followed by the
    added multibox conv layers.
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_RESNET, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if num_classes == 25: # for VOC_xlab_products dataset
            self.cfg = xl
        else:
            self.cfg = (coco, voc)[num_classes == 21]#when num_classes==21, i.e. true/[1], then voc is chosen
        self.priorbox = PriorBox(self.cfg)
        # just create an object above, but need to call forward() to return prior boxes coords
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)# ModuleList allows a list of nn.Module so that you can get it by index one by one
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])#loc conv layer
        self.conf = nn.ModuleList(head[1])#conf conv layer

        #if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, self.cfg, 200, 0.01, 0.45)

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.
        """
        sources = list()# used for storing output of chosen layers, just concat them together
        loc = list()
        conf = list()

        # apply resnet up to the last bottlneck module in stage 2, index = 10
        for k in range(11):
            x = self.base[k](x)# use resnet[k] as a function because it is a module

        s = self.L2Norm(x)
        sources.append(s)

        # apply resnet up to the last module avg pool, right before the fc1000
        # fc1000 layer has been excluded
        for k in range(11, len(self.base)):
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
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    # used for loading weights for the WHOLE SSD network (i.e. including base net), record is stored in .pth or .pkl file
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
    # Extra layers added to VGG for feature scaling
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

# this function will include above two functions for vgg base net and extra layers
def vgg_multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    # dimension for -1 or -2 is same
    vgg_source = [21, -2]# the output want to get from vgg, only conv2d layers have out_channels
    for k, v in enumerate(vgg_source):
        # Conv2d (in_channels, out_channels, kernel_size, stride, padding)
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def resnet_multibox(resnet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    resnet_source = [10, -2]#pending
    for k, v in enumerate(resnet_source):
        # Conv2d (in_channels, out_channels, kernel_size, stride, padding)
        loc_layers += [nn.Conv2d(resnet[v].out_channels(),
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(resnet[v].out_channels(),
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):# start k from 2
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return resnet, extra_layers, (loc_layers, conf_layers)

# this is the dict based on which to build the vgg and extras layers one by one
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21, base='vgg'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # str(size) will change 300 to '300'
    if base =='resnet':
        base_, extras_, head_ = resnet_multibox(resnet(),
                                         add_extras(extras[str(size)], 2048),
                                         mbox[str(size)], num_classes)
        return SSD_RESNET(phase, size, base_, extras_, head_, num_classes)
    else:
        base_, extras_, head_ = vgg_multibox(vgg(vgg_base[str(size)], 3),
                                         add_extras(extras[str(size)], 1024),
                                         mbox[str(size)], num_classes)
        return SSD_VGG(phase, size, base_, extras_, head_, num_classes)
