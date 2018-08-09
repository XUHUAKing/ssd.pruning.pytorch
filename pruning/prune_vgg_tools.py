'''
    This file contains functions for pruning vgg-like model in layer level
    1. prune_conv_layer (vgg: conv layers)

    Author: xuhuahuang as intern in YouTu 07/2018
'''
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import sys
import numpy as np

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        # layers and indexes store new layers used to update old layers
        return layers[indexes.index(i)]
    # if i not in indexes, use old layers
    return model[i]

'''
--------------------------------------------------------------------------------
     1. Prune conv layers in vgg without/with BN (only support layers stored in model.base for now)
     Args:
        model: model for pruning
        layer_index: index the pruned layer's location within model
        cut_ratio: the ratio of filters you want to prune from this layer (e.g. 20% - cut 20% lowest weights layers)
     Adapted from: https://github.com/jacobgil/pytorch-pruning
'''
def prune_conv_layer(model, layer_index, cut_ratio=0.2, use_bn = False):
    _, conv = list(model.base._modules.items())[layer_index]
    if use_bn:
        _, old_bn = list(model.base._modules.items())[layer_index + 1]
    next_conv = None
    offset = 1
    # search for the next conv, based on current conv with id = (layer_index, filter_index)
    while layer_index + offset <  len(model.base._modules.items()):
        res =  list(model.base._modules.items())[layer_index+offset] # name, module
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1

    if next_conv is None:
        print("No filter will be prunned for this layer (last layer)")
        return model

    num_filters = conv.weight.data.size(0) # out_channels x in_channels x 3 x 3
    # skip the layer with only one filter left
    if num_filters <= 1:
        print("No filter will be prunned for this layer (num_filters<=1)")
        return model

    cut = int(cut_ratio * num_filters)

    if cut < 1:
        print("No filter will be prunned for this layer (cut<1)")
        return model
    if (num_filters - cut) < 1:
        print("No filter will be prunned for this layer (no filter left after cutting)")
        return model

    # rank the filters within this layer and store into filter_ranks
    abs_wgt = torch.abs(conv.weight.data)
    values = \
        torch.sum(abs_wgt, dim = 1, keepdim = True).\
            sum(dim=2, keepdim = True).sum(dim=3, keepdim = True)[:, 0, 0, 0]# .data
    # Normalize the sum of weight by the filter dimensions in x 3 x 3
    values = values / (abs_wgt.size(1) * abs_wgt.size(2) * abs_wgt.size(3)) # (filter_number for this layer, 1)

    print("Ranking filters.. ")
    filters_to_prune = np.argsort(values.cpu().numpy())[:cut] # order from smallest to largest
    print("Filters that will be prunned", filters_to_prune)
    print("Pruning filters.. ")

    # the updated conv for current conv, with cut output channels being pruned
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - cut,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = conv.bias is not None) #(out_channels)

    old_weights = conv.weight.data.cpu().numpy() # (out_channels, in_channels, kernel_size[0], kernel_size[1]
    new_weights = new_conv.weight.data.cpu().numpy()

    # skip that filter's weight inside old_weights and store others into new_weights
    new_weights = np.delete(old_weights, filters_to_prune, axis = 0)
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    if conv.bias is not None: # no bias for conv layers
        bias_numpy = conv.bias.data.cpu().numpy()

        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune, axis = None)
        new_conv.bias.data = torch.from_numpy(bias).cuda()

    # BatchNorm modification TODO: Extract this function outside as a separate func.
    if use_bn:
        new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=old_bn.eps, momentum=old_bn.momentum, affine=old_bn.affine)
        # old_bn.affine == True, need to copy learning gamma and beta to new_bn
        # gamma: size = (num_features)
        old_weights = old_bn.weight.data.cpu().numpy()
        new_weights = new_bn.weight.data.cpu().numpy()
        new_weights = np.delete(old_weights, filters_to_prune)
        new_bn.weight.data = torch.from_numpy(new_weights).cuda()

        # beta: size = (num_features)
        bias_numpy = old_bn.bias.data.cpu().numpy()
        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune)
        new_bn.bias.data = torch.from_numpy(bias).cuda()

    # next_conv must exists
    next_new_conv = \
        torch.nn.Conv2d(in_channels = next_conv.in_channels - cut,\
            out_channels =  next_conv.out_channels, \
            kernel_size = next_conv.kernel_size, \
            stride = next_conv.stride,
            padding = next_conv.padding,
            dilation = next_conv.dilation,
            groups = next_conv.groups,
            bias = next_conv.bias is not None)

    old_weights = next_conv.weight.data.cpu().numpy()
    new_weights = next_new_conv.weight.data.cpu().numpy()

    new_weights = np.delete(old_weights, filters_to_prune, axis = 1)
    next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    if next_conv.bias is not None:
        next_new_conv.bias.data = next_conv.bias.data

    if use_bn:
        # BatchNorm modification
        base = torch.nn.Sequential(
                *(replace_layers(model.base, i, [layer_index, layer_index+1, layer_index+offset], \
                    [new_conv, new_bn, next_new_conv]) for i, _ in enumerate(model.base)))
        del old_bn
    else:
        # replace current layer and next_conv with new_conv and next_new_conv respectively
        base = torch.nn.Sequential(
                *(replace_layers(model.base, i, [layer_index, layer_index+offset], \
                    [new_conv, next_new_conv]) for i, _ in enumerate(model.base)))
    del model.base # delete and replace with brand new one
    del conv

    model.base = base
    message = str(100*float(cut) / num_filters) + "%"
    print("Filters prunned", str(message))

    return model
