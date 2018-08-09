'''
    This file contains functions for pruning resnet-like model in layer level
    1. prune_resconv_layer (resnet: conv layers)
    2. prune_resnet_lconv_layer (resnet: lconv means identity layer)
    3. prune_rbconv_by_indices (resnet: rbconv means right path's bottom layer)
    4. prune_rbconv_by_number (resnet: used when you prune lconv but next block/layer cannot absorb your effect)
    5. prune_ruconv1_layer (resnet: for resnet normal conv1 layers (i.e. right path's first upper layers))
    6. prune_ruconv2_layer (resnet: for resnet normal conv2 layers (i.e. right path's second upper layers))

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
from models.resnet import BasicBlock, Bottleneck

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        # layers and indexes store new layers used to update old layers
        return layers[indexes.index(i)]
    # if i not in indexes, use old layers
    return model[i]

'''
--------------------------------------------------------------------------------
     1. Prune conv layers in resnet with/without BN (only support layers stored in model.base for now)
     Args:
        model: model for pruning
        layer_index: index the pruned layer's location within model
        cut_ratio: the ratio of filters you want to prune from this layer (e.g. 20% - cut 20% lowest weights layers)
     Adapted from: https://github.com/jacobgil/pytorch-pruning
'''
def prune_resconv_layer(model, layer_index, cut_ratio=0.2, use_bn = True):
    _, conv = list(model.base._modules.items())[layer_index]
    if use_bn:
        _, old_bn = list(model.base._modules.items())[layer_index + 1]
    next_conv = None
    next_blk = None
    next_ds = None
    offset = 1
    # search for the next conv, based on current conv with id = (layer_index, filter_index)
    while layer_index + offset <  len(model.base._modules.items()):
        res =  list(model.base._modules.items())[layer_index+offset] # name, module
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            next_is_block = False
            break
        elif isinstance(res[1], (BasicBlock, Bottleneck)):
            next_is_block = True
            next_blk = res[1]
            if res[1].downsample is None:
                next_conv = res[1].conv1
                next_ds = None
            else:
                next_conv = res[1].conv1
                next_ds = res[1].downsample
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

    # next_ds exists or not is okay, no matter next_is_block is True or not
    if next_ds is not None:
        old_conv_in_next_ds = next_ds[0]
        new_conv_in_next_new_ds = \
            torch.nn.Conv2d(in_channels = old_conv_in_next_ds.in_channels - cut,\
                out_channels =  old_conv_in_next_ds.out_channels, \
                kernel_size = old_conv_in_next_ds.kernel_size, \
                stride = old_conv_in_next_ds.stride,
                padding = old_conv_in_next_ds.padding,
                dilation = old_conv_in_next_ds.dilation,
                groups = old_conv_in_next_ds.groups,
                bias = old_conv_in_next_ds.bias is not None)

        old_weights = old_conv_in_next_ds.weight.data.cpu().numpy()
        new_weights = new_conv_in_next_new_ds.weight.data.cpu().numpy()

        new_weights = np.delete(old_weights, filters_to_prune, axis = 1)
        new_conv_in_next_new_ds.weight.data = torch.from_numpy(new_weights).cuda()
        if old_conv_in_next_ds.bias is not None:
            new_conv_in_next_new_ds.bias.data = old_conv_in_next_ds.bias.data # bias won't change

        next_new_ds = torch.nn.Sequential(new_conv_in_next_new_ds, next_ds[1]) # BN keeps unchanged
    else:
        next_new_ds = None

    # next_new_ds and next_new_conv are ready now, create a next_new_block for replace_layers()
    if next_is_block: #same as next_blk is not None:
        if isinstance(next_blk, BasicBlock):
            # rely on conv1 of old block to get in_planes, out_planes, tride
            next_new_block = BasicBlock(next_blk.conv1.in_channels - cut, next_blk.conv1.out_channels, next_blk.stride, downsample = next_new_ds)
            next_new_block.conv1 = next_new_conv # only update in_channels
            next_new_block.bn1 = next_blk.bn1
            next_new_block.relu = next_blk.relu
            next_new_block.conv2 = next_blk.conv2
            next_new_block.bn2 = next_blk.bn2
        else:
            next_new_block = Bottleneck(next_blk.conv1.in_channels - cut, next_blk.conv1.out_channels, next_blk.stride, downsample = next_new_ds)
            next_new_block.conv1 = next_new_conv # only update in_channels
            next_new_block.bn1 = next_blk.bn1
            next_new_block.conv2 = next_blk.conv2
            next_new_block.bn2 = next_blk.bn2
            next_new_block.conv3 = next_blk.conv3
            next_new_block.bn3 = next_blk.bn3
            next_new_block.relu = next_blk.relu

    if not next_is_block:
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
    else:
        if use_bn:
            # BatchNorm modification
            base = torch.nn.Sequential(
                    *(replace_layers(model.base, i, [layer_index, layer_index+1, layer_index+offset], \
                        [new_conv, new_bn, next_new_block]) for i, _ in enumerate(model.base)))
            del old_bn
        else:
            # replace current layer and next_conv with new_conv and next_new_conv respectively
            base = torch.nn.Sequential(
                    *(replace_layers(model.base, i, [layer_index, layer_index+offset], \
                        [new_conv, next_new_block]) for i, _ in enumerate(model.base)))

    del model.base # delete and replace with brand new one
    del conv

    model.base = base
    message = str(100*float(cut) / num_filters) + "%"
    print("Filters prunned", str(message))

    return model

'''
--------------------------------------------------------------------------------
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
    _, blk = list(model.base._modules.items())[block_index]
    cut_indices = None

    if not use_bn:
        print("ResNet without BN is not supported for prunning")
        return cut_indices, model

    # check whether the left path has conv layer for prunning
    if blk.downsample == None:
        print("No filters will be prunned because lconv doesn't exist")
        return cut_indices, model

    if not isinstance(blk, (BasicBlock, Bottleneck)):
        print("Only support for ResNet with BasicBlock or Bottleneck defined in torchvision")
        return cut_indices, model

    # get old conv and bn on the left
    lconv = blk.downsample[0] # nn.Sequential for (lconv, lbn)
    lbn = blk.downsample[1]
    next_conv = None
    next_ds = None # if next one is a block, and this block has downsample path, you need to update both residual and downsample path
    next_blk = None
    offset = 1
    # search for the next conv, can be conv1 within next block, or a normal conv layer
    while block_index + offset <  len(model.base._modules.items()):
        res =  list(model.base._modules.items())[block_index+offset] # name, module
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            next_is_block = False
            break
        elif isinstance(res[1], (BasicBlock, Bottleneck)):
            next_is_block = True
            next_blk = res[1]
            if res[1].downsample is None:
                next_conv = res[1].conv1
                next_ds = None
            else:
                next_conv = res[1].conv1
                next_ds = res[1].downsample
            break
        offset = offset + 1

    if next_conv is None:
        print("No filters will be prunned because this is the last layer")
        return cut_indices, model

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

    # the updated conv for old lconv, with cut output channels being pruned
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

    if lconv.bias is not None:
        bias_numpy = lconv.bias.data.cpu().numpy()

        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune, axis = None)
        new_conv.bias.data = torch.from_numpy(bias).cuda()

    # new BN layer after new_conv
    new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=lbn.eps, momentum=lbn.momentum, affine=lbn.affine)
    # old_bn.affine == True, need to copy learnable gamma and beta to new_bn
    # gamma: size = (num_features)
    old_weights = lbn.weight.data.cpu().numpy()
    new_weights = new_bn.weight.data.cpu().numpy()
    new_weights = np.delete(old_weights, filters_to_prune)
    new_bn.weight.data = torch.from_numpy(new_weights).cuda()

    # beta: size = (num_features)
    bias_numpy = lbn.bias.data.cpu().numpy()
    # change size to (out_channels - cut)
    bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
    bias = np.delete(bias_numpy, filters_to_prune)
    new_bn.bias.data = torch.from_numpy(bias).cuda()

    # next_conv exists, update next_conv or next_conv of a block or next_conv+next_ds of a block
    # next_conv must exist
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
        next_new_conv.bias.data = next_conv.bias.data # bias won't change

    # next_ds exists or not is okay, no matter next_is_block is True or not
    if next_ds is not None:
        old_conv_in_next_ds = next_ds[0]
        new_conv_in_next_new_ds = \
            torch.nn.Conv2d(in_channels = old_conv_in_next_ds.in_channels - cut,\
                out_channels =  old_conv_in_next_ds.out_channels, \
                kernel_size = old_conv_in_next_ds.kernel_size, \
                stride = old_conv_in_next_ds.stride,
                padding = old_conv_in_next_ds.padding,
                dilation = old_conv_in_next_ds.dilation,
                groups = old_conv_in_next_ds.groups,
                bias = old_conv_in_next_ds.bias is not None)

        old_weights = old_conv_in_next_ds.weight.data.cpu().numpy()
        new_weights = new_conv_in_next_new_ds.weight.data.cpu().numpy()

        new_weights = np.delete(old_weights, filters_to_prune, axis = 1)
        new_conv_in_next_new_ds.weight.data = torch.from_numpy(new_weights).cuda()
        if old_conv_in_next_ds.bias is not None:
            new_conv_in_next_new_ds.bias.data = old_conv_in_next_ds.bias.data # bias won't change

        next_new_ds = torch.nn.Sequential(new_conv_in_next_new_ds, next_ds[1]) # BN keeps unchanged
    else:
        next_new_ds = None

    # next_new_ds and next_new_conv are ready now, create a next_new_block for replace_layers()
    if next_is_block: #same as next_blk is not None:
        if isinstance(next_blk, BasicBlock):
            # rely on conv1 of old block to get in_planes, out_planes, tride
            next_new_block = BasicBlock(next_blk.conv1.in_channels - cut, next_blk.conv1.out_channels, next_blk.stride, downsample = next_new_ds)
            next_new_block.conv1 = next_new_conv # only update in_channels
            next_new_block.bn1 = next_blk.bn1
            next_new_block.relu = next_blk.relu
            next_new_block.conv2 = next_blk.conv2
            next_new_block.bn2 = next_blk.bn2
        else:
            next_new_block = Bottleneck(next_blk.conv1.in_channels - cut, next_blk.conv1.out_channels, next_blk.stride, downsample = next_new_ds)
            next_new_block.conv1 = next_new_conv # only update in_channels
            next_new_block.bn1 = next_blk.bn1
            next_new_block.conv2 = next_blk.conv2
            next_new_block.bn2 = next_blk.bn2
            next_new_block.conv3 = next_blk.conv3
            next_new_block.bn3 = next_blk.bn3
            next_new_block.relu = next_blk.relu

    # replace
    # update current left conv + left BN layer, have BN by default
    new_ds = torch.nn.Sequential(
        *(replace_layers(blk.downsample, i, [0, 1], \
            [new_conv, new_bn]) for i, _ in enumerate(blk.downsample)))

    # delete current and replace with a brand new BLOCK
    if isinstance(blk, BasicBlock):
        # rely on conv1 of old block to get in_planes, out_planes, tride
        new_blk = BasicBlock(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = new_ds)
        # keep all layers in residual path unchanged tempararily
        new_blk.conv1 = blk.conv1
        new_blk.bn1 = blk.bn1
        new_blk.relu = blk.relu
        new_blk.conv2 = blk.conv2
        new_blk.bn2 = blk.bn2
    else:
        new_blk = Bottleneck(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = new_ds)
        # keep all layers in residual path unchanged tempararily
        new_blk.conv1 = blk.conv1
        new_blk.bn1 = blk.bn1
        new_blk.conv2 = blk.conv2
        new_blk.bn2 = blk.bn2
        new_blk.conv3 = blk.conv3
        new_blk.bn3 = blk.bn3
        new_blk.relu = blk.relu

    # now new_blk is ready, it can act as a layer and replace old blk with replace_layers()
    # update next_new_conv/next_new_block together
    if not next_is_block:
        # next_conv is a normal conv layer
        base = torch.nn.Sequential(
                *(replace_layers(model.base, i, [block_index, block_index+offset], \
                    [new_blk, next_new_conv]) for i, _ in enumerate(model.base)))
    else:
        # next_conv is a block, need to replace it with a new blk with updated conv1, and downsample if necessary
        base = torch.nn.Sequential(
                *(replace_layers(model.base, i, [block_index, block_index+offset], \
                    [new_blk, next_new_block]) for i, _ in enumerate(model.base)))

    # delete and replace with brand new one
    del model.base # delete the things pointed by pointer
    model.base = base

    cut_indices = filters_to_prune
    message = str(100*float(cut) / num_filters) + "%"
    print("Filters prunned", str(message))

    return cut_indices, model

'''
--------------------------------------------------------------------------------
    3. Prune residual conv layer, the one at the bottom of residual side with/without BN
    (*Note: MUST call this after you prune identity path with downsample, the size won't fit because upper functions only update left path)
    Args:
        block_index: the BasicBlock or Bottleneck Block this layer locates
        filters_to_prune: the filters' indices waiting for being pruned
        use_bn: use Batch Norm or not
'''
def prune_rbconv_by_indices(model, block_index, filters_to_prune, use_bn = True):
    _, blk = list(model.base._modules.items())[block_index]

    if not use_bn:
        print("ResNet without BN is not supported for prunning")
        return model

    # check whether the left path has conv layer for prunning
    if blk.downsample == None:
        print("Only support pruning for rbconv after lconv was pruned")
        return model

    if not isinstance(blk, (BasicBlock, Bottleneck)):
        print("Only support for ResNet with BasicBlock or Bottleneck defined in torchvision")
        return model

    if isinstance(blk, BasicBlock):
        # when it is BasicBlock, the rbconv is conv2, and its bn is bn2
        conv = blk.conv2
        bn = blk.bn2
        # only need to update itself, no need to care about others such as next_ds/next_conv
        new_conv = \
            torch.nn.Conv2d(in_channels = conv.in_channels, \
                out_channels = conv.out_channels - len(filters_to_prune),
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

        if conv.bias is not None:
            bias_numpy = conv.bias.data.cpu().numpy()

            # change size to (out_channels - cut)
            bias = np.zeros(shape = (bias_numpy.shape[0] - len(filters_to_prune)), dtype = np.float32)
            bias = np.delete(bias_numpy, filters_to_prune, axis = None)
            new_conv.bias.data = torch.from_numpy(bias).cuda()

        # new BN layer after new_conv
        new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
        # old_bn.affine == True, need to copy learnable gamma and beta to new_bn
        # gamma: size = (num_features)
        old_weights = bn.weight.data.cpu().numpy()
        new_weights = new_bn.weight.data.cpu().numpy()
        new_weights = np.delete(old_weights, filters_to_prune)
        new_bn.weight.data = torch.from_numpy(new_weights).cuda()

        # beta: size = (num_features)
        bias_numpy = bn.bias.data.cpu().numpy()
        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - len(filters_to_prune)), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune)
        new_bn.bias.data = torch.from_numpy(bias).cuda()

        # replace with new block
        new_blk = BasicBlock(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = blk.downsample)
        # keep all layers in residual path unchanged tempararily
        new_blk.conv1 = blk.conv1
        new_blk.bn1 = blk.bn1
        new_blk.relu = blk.relu
        new_blk.conv2 = new_conv # update with new conv
        new_blk.bn2 = new_bn # update with new bn

    else:
        # when it is Bottleneck, the rbconv is conv3, and its bn is bn3
        conv = blk.conv3
        bn = blk.bn3
        # only need to update itself, no need to care about others such as next_ds/next_conv
        new_conv = \
            torch.nn.Conv2d(in_channels = conv.in_channels, \
                out_channels = conv.out_channels - len(filters_to_prune),
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

        if conv.bias is not None:
            bias_numpy = conv.bias.data.cpu().numpy()

            # change size to (out_channels - cut)
            bias = np.zeros(shape = (bias_numpy.shape[0] - len(filters_to_prune)), dtype = np.float32)
            bias = np.delete(bias_numpy, filters_to_prune, axis = None)
            new_conv.bias.data = torch.from_numpy(bias).cuda()

        # new BN layer after new_conv
        new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
        # old_bn.affine == True, need to copy learnable gamma and beta to new_bn
        # gamma: size = (num_features)
        old_weights = bn.weight.data.cpu().numpy()
        new_weights = new_bn.weight.data.cpu().numpy()
        new_weights = np.delete(old_weights, filters_to_prune)
        new_bn.weight.data = torch.from_numpy(new_weights).cuda()

        # beta: size = (num_features)
        bias_numpy = bn.bias.data.cpu().numpy()
        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - len(filters_to_prune)), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune)
        new_bn.bias.data = torch.from_numpy(bias).cuda()

        # replace with new block
        new_blk = Bottleneck(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = blk.downsample)
        # keep all layers in residual path unchanged tempararily
        new_blk.conv1 = blk.conv1
        new_blk.bn1 = blk.bn1
        new_blk.conv2 = blk.conv2
        new_blk.bn2 = blk.bn2
        new_blk.conv3 = new_conv
        new_blk.bn3 = new_bn
        new_blk.relu = blk.relu

    base = torch.nn.Sequential(
            *(replace_layers(model.base, i, [block_index], \
                [new_blk]) for i, _ in enumerate(model.base)))

    # delete and replace
    del model.base
    model.base = base
    print("Filters prunned:", filters_to_prune)

    return model

'''
--------------------------------------------------------------------------------
    4. Prune residual conv layer, the one at the bottom of residual side with/without BN, based on its own weights
    (*Note: MUST call this when you prune lconv layer,
            the immediate following block/conv cannot absorb your effect due to its empty left path)
    Args:
        block_index: the BasicBlock or Bottleneck Block this layer locates
        num_cut: the number of filters waiting for being pruned
        use_bn: use Batch Norm or not
'''
def prune_rbconv_by_number(model, block_index, num_cut, use_bn = True):


'''
--------------------------------------------------------------------------------
    5. Prune normal residual conv layer, the FRIST one at the upper of residual side with/without BN
    Args:
        block_index: the BasicBlock or Bottleneck Block this layer locates
        cut_ratio: the ratio of filters pruned from conv1 (and conv2 if Bottleneck)
        use_bn: use Batch Norm or not
'''
def prune_ruconv1_layer(model, block_index, cut_ratio=0.2, use_bn = True):
    _, blk = list(model.base._modules.items())[block_index]

    if not use_bn:
        print("ResNet without BN is not supported for prunning")
        return model

    if not isinstance(blk, (BasicBlock, Bottleneck)):
        print("Conv1 only for ResNet with BasicBlock or Bottleneck defined in torchvision")
        return model
    # cut conv1, and next conv is conv2
    conv = blk.conv1
    bn = blk.bn1
    next_conv = blk.conv2

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

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune, axis = None)
        new_conv.bias.data = torch.from_numpy(bias).cuda() # new conv1

    # BatchNorm layer
    new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
    # gamma: size = (num_features)
    old_weights = bn.weight.data.cpu().numpy()
    new_weights = bn.weight.data.cpu().numpy()
    new_weights = np.delete(old_weights, filters_to_prune)
    new_bn.weight.data = torch.from_numpy(new_weights).cuda()

    # beta: size = (num_features)
    bias_numpy = bn.bias.data.cpu().numpy()
    # change size to (out_channels - cut)
    bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
    bias = np.delete(bias_numpy, filters_to_prune)
    new_bn.bias.data = torch.from_numpy(bias).cuda() # new bn1

    # new conv for next_conv
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
        next_new_conv.bias.data = next_conv.bias.data # new conv2

    # replace with new block
    if isinstance(blk, BasicBlock):
        new_blk = BasicBlock(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = blk.downsample)
        # keep all layers in residual path unchanged tempararily
        new_blk.conv1 = new_conv
        new_blk.bn1 = new_bn
        new_blk.relu = blk.relu
        new_blk.conv2 = next_new_conv # update with new conv
        new_blk.bn2 = blk.bn2 # update with new bn
    else:
        new_blk = Bottleneck(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = blk.downsample)
        # keep all layers in residual path unchanged tempararily
        new_blk.conv1 = new_conv
        new_blk.bn1 = new_bn
        new_blk.conv2 = next_new_conv
        new_blk.bn2 = blk.bn2
        new_blk.conv3 = blk.conv3
        new_blk.bn3 = blk.bn3
        new_blk.relu = blk.relu

    base = torch.nn.Sequential(
            *(replace_layers(model.base, i, [block_index], \
                [new_blk]) for i, _ in enumerate(model.base)))

    # delete and replace
    del model.base
    model.base = base
    print("Filters prunned:", filters_to_prune)

    return model

'''
--------------------------------------------------------------------------------
    6. Prune normal residual conv layer, the SECOND one at the upper of residual side with/without BN
       (*for Bottleneck only)
    Args:
        block_index: the BasicBlock or Bottleneck Block this layer locates
        cut_ratio: the ratio of filters pruned from conv1 (and conv2 if Bottleneck)
        use_bn: use Batch Norm or not
'''
def prune_ruconv2_layer(model, block_index, cut_ratio=0.2, use_bn = True):
    _, blk = list(model.base._modules.items())[block_index]

    if not use_bn:
        print("ResNet without BN is not supported for prunning")
        return model

    if not isinstance(blk, Bottleneck):
        print("Conv2 only for ResNet with Bottleneck defined in torchvision")
        return model
    # cut conv1, and next conv is conv2
    conv = blk.conv2
    bn = blk.bn2
    next_conv = blk.conv3

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

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()

        # change size to (out_channels - cut)
        bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
        bias = np.delete(bias_numpy, filters_to_prune, axis = None)
        new_conv.bias.data = torch.from_numpy(bias).cuda() # new conv2

    # BatchNorm layer
    new_bn = torch.nn.BatchNorm2d(num_features=new_conv.out_channels, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
    # gamma: size = (num_features)
    old_weights = bn.weight.data.cpu().numpy()
    new_weights = bn.weight.data.cpu().numpy()
    new_weights = np.delete(old_weights, filters_to_prune)
    new_bn.weight.data = torch.from_numpy(new_weights).cuda()

    # beta: size = (num_features)
    bias_numpy = bn.bias.data.cpu().numpy()
    # change size to (out_channels - cut)
    bias = np.zeros(shape = (bias_numpy.shape[0] - cut), dtype = np.float32)
    bias = np.delete(bias_numpy, filters_to_prune)
    new_bn.bias.data = torch.from_numpy(bias).cuda() # new bn2

    # new conv for next_conv
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
        next_new_conv.bias.data = next_conv.bias.data # new conv3

    # replace with new block
    new_blk = Bottleneck(blk.conv1.in_channels, blk.conv1.out_channels, blk.stride, downsample = blk.downsample)
    # keep all layers in residual path unchanged tempararily
    new_blk.conv1 = blk.conv1
    new_blk.bn1 = blk.bn1
    new_blk.conv2 = new_conv
    new_blk.bn2 = new_bn
    new_blk.conv3 = next_new_conv
    new_blk.bn3 = blk.bn3
    new_blk.relu = blk.relu

    base = torch.nn.Sequential(
            *(replace_layers(model.base, i, [block_index], \
                [new_blk]) for i, _ in enumerate(model.base)))

    # delete and replace
    del model.base
    model.base = base
    print("Filters prunned:", filters_to_prune)

    return model
