# Improved Single Shot MultiBox Object Detector, RefineDet and Network Pruning, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of:
- [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, etc.
- Variants of SSD with Resnet-50/MobileNetv1/MobileNetv2 backbones.
- [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897) from the 2017 paper by Shifeng Zhang, etc.
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) from the 2017 paper by Hao Li, etc.
- Support model training and evaluation on various datasets including VOC/XLab/WEISHI/COCO.  

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#train-and-test'>Train and Test</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#prune-and-finetune'>Prune and Finetune</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Clone this repository.
  * Note: We currently only support Python 3+ and PyTorch 0.3.
  * Note: Can directly use image in Tencent Docker `youtu/akuxcwchen_pytorch:3.0` for environment setup.
- Then download the dataset by following the [instructions](#datasets) below.
- We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training!
  * To use Visdom in the browser:
  ```Shell
  # First install Python server and client
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training and evaluation, [COCO](http://mscoco.org/) is not supported yet.

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

Please refer to `config.py` file (path: ssd.pytorch.tencent/data) and remember to update dataset root if necessary. Please also note that dataset root for VALIDATION should be written within `config.py`, while dataset root for TRAINING can be updated through args during execution of a program. 

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
# this dataset has existed in /cephfs/share/data/VOCdevkit in Tencent server
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
# this dataset has existed in /cephfs/share/data/VOCdevkit in Tencent server
```

### XL Dataset
This dataset is VOC-like dataset produced by X-Lab
```Shell
# this dataset has existed in /cephfs/share/data/VOC_xlab_products in Tencent server
```
### WEISHI Dataset
This dataset has an one-to-one matching format inside a jpg-xml file. 
```shell
# this dataset has existed in /cephfs/share/data/weishi_xh in Tencent server
```

### COCO (Not supported now but can refer to pycocotools API)
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
# this dataset has existed in /cephfs/share/data/coco_xy in Tencent server
```

## Train and Test
### Model Set-up
- All required backbone weights have existed in `ssd.pytorch.tencent/weights` dir. They are modified versions of original model (Resnet-50, VGG, MobileNet v1, MobileNet v2) fitting our own model design.

```Shell
# navigate them by:
mkdir weights
cd weights
```
- To make backbone preloading more convenient, we turn all backbone models into class object inheriting `nn.Module` in PyTorch. 
  * Resnet: `ssd.pytorch.tencent/models/resnet.py`
  * VGG for SSD + RefineDet: `ssd.pytorch.tencent/models/vgg.py`
  * MobileNet v1: `ssd.pytorch.tencent/models/mobilenetv1.py`
  * MobileNet v2: `ssd.pytorch.tencent/models/mobilenetv2.py`
 
Based on this design, all backbone layers returned by functions in `backbones.py` have already stored pretrained weights.

### Training SSD (Resnet/VGG/MobileNetv1/MobileNetv2)
- To train SSD using the train script simply specify the parameters listed in `train_test_vrmSSD.py` as a flag or manually change them.

```Shell
#Use VOC dataset by default
#Train + Test SSD model with vgg backbone
python3 train_test_vrmSSD.py --evaluate True # testing while training
python3 train_test_vrmSSD.py # only training

#Train + Test SSD model with resnet backbone
python3 train_test_vrmSSD.py --use_res --evaluate True # testing while training
python3 train_test_vrmSSD.py --use_res # only training

#Train + Test SSD model with mobilev1 backbone
python3 train_test_vrmSSD.py --use_m1 --evaluate True # testing while training
python3 train_test_vrmSSD.py --use_m1 # only training

#Train + Test SSD model with mobilev2 backbone
python3 train_test_vrmSSD.py --use_m2 --evaluate True # testing while training
python3 train_test_vrmSSD.py --use_m2 # only training

#Use WEISHI dataset
--dataset WEISHI --dataset_root _path_for_WEISHI_ROOT --jpg_xml_path _path_to_your_jpg_xml_txt

#Use XL dataset
--dataset XL --dataset_root _path_for_XL_ROOT
```
### Training RefineDet (VGG)
- To train RefineDet using the train script simply specify the parameters listed in `train_test_refineDet.py` as flag or manually change them.
```Shell
#Use VOC dataset by default
#Train + Test refineDet model
python3 train_test_refineDet.py --evaluate True #testing while training
python3 train_test_refineDet.py #only training

#Use WEISHI dataset
--dataset WEISHI --dataset_root _path_for_WEISHI_ROOT --jpg_xml_path _path_of_your_jpg_xml

#Use XL dataset
--dataset XL --dataset_root _path_for_XL_ROOT
```
- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `--resume` for options)

## Evaluation
To evaluate a trained network or checkpoint on VOC or VOC-like dataset only.
### Evaluate SSD (Resnet/VGG/MobileNetv1/MobileNetv2)
Use `eval_voc_vrmSSD.py` to evaluate.
```Shell
#Model evaluation on VOC for vggSSD separately
python3 eval_voc_vrmSSD.py --trained_model weights/_your_trained_SSD_model_.pth

#Model evaluation on VOC for resnetSSD separately
python3 eval_voc_vrmSSD.py --use_res --trained_model weights/_your_trained_SSD_model_.pth

#Model evaluation on VOC for mobileSSD v1 separately
python3 eval_voc_vrmSSD.py --use_m1 --trained_model weights/_your_trained_SSD_model_.pth

#Model evaluation on VOC for mobileSSD v2 separately
python3 eval_voc_vrmSSD.py --use_m2 --trained_model weights/_your_trained_SSD_model_.pth

#Take care of different versions of .pth file, can be solved by changing state_dict
```  
### Evaluate RefineDet (VGG)
Use `eval_voc_refineDet.py` to evaluate.
```Shell
#Model evaluation on VOC for refineDet separately
python3 eval_voc_refineDet.py --trained_model weights/_your_trained_refineDet_model_.pth

#Take care of different versions of .pth file, can be solved by changing state_dict
```

For other datasets, please refer to Test part in train_test files, and extract the test_net() function correspondingly.

<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Prune and Finetune
### Prune
#### Following files are for maginitude-based filter pruning purpose:
- `prune_weights_refineDet.py`
- `prune_weights_resnetSSD.py`
- `prune_weights_vggSSD.py`

```Shell
#Use absolute weights-based criterion for filter pruning on refineDet(vgg)
python3 prune_weights_refineDet.py --trained_model weights/_your_trained_model_.pth

#Use absolute weights-based criterion for filter pruning on vggSSD
python3 prune_weights_vggSSD.py --trained_model weights/_your_trained_model_.pth

#Use absolute weights-based criterion for filter pruning on resnetSSD (resnet50)
python3 prune_weights_resnetSSD.py --trained_model weights/_your_trained_model_.pth
#**Note**
#Due to the limitation of PyTorch, if you really need to prune left path conv layer,
#after call this file, please use prune_rbconv_by_number() MANUALLY to prune all following right bottom layers affected by your pruning

```
The way of loading trained model (first time) and finetuned model (> 2 times) are different.
Please change the following codes within `prune_weights_**.py` files correspondingly.
```Shell
# ------------------------------------------- 1st prune: load model from state_dict
model = build_ssd('train', cfg, cfg['min_dim'], cfg['num_classes'], base='resnet').cuda()
state_dict = torch.load(args.trained_model)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7] # head = k[:4]
    if head == 'module.': # head == 'vgg.', module. is due to DataParellel
        name = k[7:]  # name = 'base.' + k[4:]
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
#model.load_state_dict(torch.load(args.trained_model))

# ------------------------------------------- >= 2nd prune: load model from previous pruning
model = torch.load(args.trained_model).cuda()
```

### Finetune
#### Following files are for fintuning purpose after pruning/previous finetuing:
- `finetune_vggresSSD.py`
- `finetune_refineDet.py`
```Shell
#Finetune prunned model vggSSD (Train/Test on VOC)
python3 finetune_vggresSSD.py --pruned_model prunes/_your_prunned_model_ --lr x --epoch y

#Finetune prunned model resnetSSD (Train/Test on VOC)
python3 finetune_vggresSSD.py --use_res --pruned_model prunes/_your_prunned_model_ --lr x --epoch y
    
#Finetune prunned model refineDet(vgg) (Train/Test on VOC)
python3 finetune_refineDet.py --pruned_model prunes/_your_prunned_model --lr x --epoch y
```
## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |

##### FPS
**GTX 1060:** ~45.45 FPS

## Demos

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- For other pre-trained network with different backbones on different dataset, please contact me.
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325)
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    2. If using [pip](https://pypi.python.org/pypi/pip):

```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

## TODO
The following to-do list, which hope to complete in the near future
- Still to come:
  * [x] Fix mistake for implemenation of top_k/max_per_image
  * [ ] Fix learning rate to 0.98 decay every epoch and 1/10 when mAP is stable
  * [ ] Support for the MS COCO dataset
  * [ ] Support for SSD512 and RefineDet512 training and testing

## Authors

* [**xuhuahuang**](https://github.com/XUHUAKing) as intern in Tencent YouTu Lab 07/2018 

Thanks [**Max deGroot**](https://github.com/amdegroot) and [**Ellis Brown**](http://github.com/ellisbrown) because this work is built on their original implementation for SSD in pytorch.


## References
- Jaco. "PyTorch Implementation of [1611.06440] Pruning Convolutional Neural Networks for Resource Efficient Inference." https://github.com/jacobgil/pytorch-pruningReferences
- Implementation of Variants of SSD Model. https://github.com/lzx1413/PytorchSSD
- Useful links of explanation for Mobilenet v2. http://machinethink.net/blog/mobilenet-v2/
- Useful links of explanation for Mobilenet v1. http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
