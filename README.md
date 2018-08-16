# Improved Single Shot MultiBox Object Detector, RefineDet and Network Pruning, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of:
- [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, etc.
- Variants of SSD with Resnet-50/MobileNetv1/MobileNetv2 backbones.
- [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897) from the 2017 paper by Shifeng Zhang, etc.
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) from the 2017 paper by Hao Li, etc.
- Support model training and evaluation on various datasets including VOC/XLab/WEISHI/COCO.  

<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#pruning-and-fintune'>Pruning and Finetune</a>
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

Please refer to config.py file (path: ssd.pytorch.tencent/data) and remember to update dataset root if necessary. Please also note that dataset root for VALIDATION should be written within config.py, while dataset root for TRAINING can be updated through args during execution of a program. 


### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
# this dataset has existed in /cephfs/share/data/coco_xy in Tencent server
```

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
```Shell
# this dataset has existed in /cephfs/share/data/VOC_xlab_products in Tencent server
```
### WEISHI Dataset
```shell
# this dataset has existed in /cephfs/share/data/weishi_xh in Tencent server
```
## Training
### Set Up
- All required backbone weights have existed in `ssd.pytorch.tencent/weights` dir. They are modified versions of original model (Resnet-50, VGG, MobileNet v1, MobileNet v2).

```Shell
mkdir weights
cd weights
```
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

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Pruning and Finetune

### Pruning
To prune a trained network (first time):

To prune a finetuned network (>2 times):

### Finetune

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
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [ ] Support for the MS COCO dataset
  * [ ] Support for the WEISHI dataset
  * [ ] Change backbone of refineDet to resnet
  * [ ] Support for SSD512 training and testing

## Authors

* [**Xuhua HUANG**](https://github.com/XUHUAKing)
* [**Max deGroot**](https://github.com/amdegroot)
* [**Ellis Brown**](http://github.com/ellisbrown)

***Note:*** Unfortunately, this is just a hobby of ours and not a full-time job, so we'll do our best to keep things up to date, but no guarantees.  That being said, thanks to everyone for your continued help and feedback as it is really appreciated. We will try to address everything as soon as possible.

## References
- Jaco. "PyTorch Implementation of [1611.06440] Pruning Convolutional Neural Networks for Resource Efficient Inference." https://github.com/jacobgil/pytorch-pruningReferences
- Implementation of Variants of SSD Model. https://github.com/lzx1413/PytorchSSD
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo):
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow)
