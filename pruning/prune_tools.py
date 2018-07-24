'''
    This file contains functions for pruning in layer level
    1. prune_conv_layer (for resnet normal conv layers (i.e. right path's upper layers) and vgg conv layers)
    2. prune_resnet_lconv_layer (lconv means identity layer)
    3. prune_rbconv_by_indices_no_bn (rbconv means right path's bottom layer)
    4. prune_rbconv_by_indices_with_bn
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
     1. Prune conv layers without/with BN (only support layers stored in model.base for now)
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

    bias_numpy = conv.bias.data.cpu().numpy()

    # change size to (out_channels - cut)
    bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
    bias = np.delete(bias_numpy, filters_to_prune, axis = None)
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    # BatchNorm modification
    if use_bn:
        new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=old_bn.eps, momentum=old_bn.momentum, affine=old_bn.affine)

    # next_conv exists
    if not next_conv is None:
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

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        if use_bn:
            # BatchNorm modification
            base = torch.nn.Sequential(
                    *(replace_layers(model.base, i, [layer_index, layer_index+1, layer_index+offset], \
                        [new_conv, new_bn, next_new_conv]) for i, _ in enumerate(model.base)))
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

    else:
        print("Pruning the last conv layer, error!")

    return model

'''
    2. Prune identity conv layers without/with BN in a resnet block
    (*Note: NOT used for normal layer, the 'layer' here must locate inside a block indexed by block_index)
    Args:
        block_index: a block also named as a 'layer' in torchvision implementation, locate lconv layer
        *Note:
        The index criteria based on 'one single block' unit, which means 1 index represents 1 BasicBlock/Bottleneck, instead of one layer (3-6 blocks)
    Return:
        cut_indices: the filters_to_prune in this layer, will be used in function 5.
'''
def prune_resnet_lconv_layer(model, block_index, cut_ratio=0.2, use_bn = True):
    _, blk = model.base._modules.items()[block_index]
    if use_bn:
        _, old_bn =
    cut_indices = np.asarray([])

    if not use_bn:
        print("ResNet without BN is not supported for prunning")
        return cut_indices, model

    # check whether the left path has conv layer for prunning
    if blk.downsample == None:
        print("No filters will be prunned because lconv doesn't exist")
        return cut_indices, model

    lconv = blk.downsample[0] # nn.Sequential
    lbn = blk.downsample[1]
    next_conv = None
    offset = 1
    # search for the next conv, can be conv1 within next block, or a normal conv layer
    while block_index + offset <  len(model.base._modules.items()):
        res =  model.base._modules.items()[block_index+offset] # name, module
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            next_is_block = False
            break
        elif isinstance(res[1], (BasicBlock, Bottleneck)):
            next_conv = res[1].conv1
            next_is_block = True
            break
        offset = offset + 1

    num_filters = lconv.weight.data.size(0) # out_channels x in_channels x 3 x 3
    # skip the layer with only one filter left
    if num_filters <= 1:
        print("No filter will be prunned for this layer (num_filters<=1)")
        return cut_indices, model

    cut = int(cut_ratio * num_filters)

    if cut < 1:
        print("No filter will be prunned for this layer (cut<1)")
        return cut_indices, model
    if (num_filters - cut) < 1:
        print("No filter will be prunned for this layer (no filter left after cutting)")
        return cut_indices, model

    # rank the filters within this layer and store into filter_ranks
    abs_wgt = torch.abs(lconv.weight.data)
    values = \
        torch.sum(abs_wgt, dim = 1, keepdim = True).\
            sum(dim=2, keepdim = True).sum(dim=3, keepdim = True)[:, 0, 0, 0]# .data
    # Normalize the sum of weight by the filter dimensions in x 3 x 3
    values = values / (abs_wgt.size(1) * abs_wgt.size(2) * abs_wgt.size(3)) # (filter_number for this layer, 1)

    print("Ranking filters.. ")
    filters_to_prune = np.argsort(values.cpu().numpy())[:cut] # order from smallest to largest
    print("Filters that will be prunned", filters_to_prune)
    print("Pruning filters.. ")

    # the updated conv for current lconv, with cut output channels being pruned
    new_conv = \
        torch.nn.Conv2d(in_channels = lconv.in_channels, \
            out_channels = lconv.out_channels - cut,
            kernel_size = lconv.kernel_size, \
            stride = lconv.stride,
            padding = lconv.padding,
            dilation = lconv.dilation,
            groups = lconv.groups,
            bias = lconv.bias is not None) #(out_channels)

    old_weights = lconv.weight.data.cpu().numpy() # (out_channels, in_channels, kernel_size[0], kernel_size[1]
    new_weights = new_conv.weight.data.cpu().numpy()

    # skip that filter's weight inside old_weights and store others into new_weights
    new_weights = np.delete(old_weights, filters_to_prune, axis = 0)
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = lconv.bias.data.cpu().numpy()

    # change size to (out_channels - cut)
    bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
    bias = np.delete(bias_numpy, filters_to_prune, axis = None)
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    # new BN layer after new_conv
    if use_bn:
        new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=lbn.eps, momentum=lbn.momentum, affine=lbn.affine)

    # next_conv exists, update next_conv
    if not next_conv is None:
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

        next_new_conv.bias.data = next_conv.bias.data

    # replace the old layers
    if not next_conv is None:
        if use_bn:
            # update current conv
			new_ds = torch.nn.Sequential(
				*(replace_layers(model.base._modules.items()[block_index][1].downsample, i, [0, 1], \
					[new_conv, new_bn]) for i, _ in enumerate(model.base._modules.items()[block_index][1].downsample)))
			model.base._modules.items()[block_index][1].downsample = new_ds
            # update next_conv
            if not next_is_block:
                # next_conv is a normal conv layer
                base = torch.nn.Sequential(
                        *(replace_layers(model.base, i, [block_index+offset], \
                            [next_new_conv]) for i, _ in enumerate(model.base)))
        else:
            print("ResNet without BN is not supported for prunning")

        del model.base # delete and replace with brand new one

        model.base = base
        cut_indices = filters_to_prune
        message = str(100*float(cut) / num_filters) + "%"
        print("Filters prunned", str(message))

    else:
        print("Pruning the last conv layer, error!")

    return cut_indices, model

'''
    3. Prune residual conv layer, the one at the bottom of residual side without BN
    Args:
        block_index: the BasicBlock or Bottleneck Block this layer locates
        filters_to_prune: the filters' indices waiting for being pruned
'''
def prune_rbconv_by_indices_no_bn(model, block_index, filters_to_prune):
    pass

'''
    4. Prune residual conv layer, the one at the bottom of residual side with BN
    Args:
        block_index: the BasicBlock or Bottleneck Block this layer locates
        filters_to_prune: the filters' indices waiting for being pruned
'''
def prune_rbconv_by_indices_with_bn(model, block_index, filters_to_prune):
    pass
