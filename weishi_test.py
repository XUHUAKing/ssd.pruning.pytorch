from __future__ import print_function

from data import *
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WEISHI_CLASSES as labelmap #WEISHI_CLASSES is a global variable, storing information about weishi dataset now
from PIL import Image
from data import BaseTransform, WEISHI_CLASSES
import torch.utils.data as data
from models.SSD_vggres import build_ssd

if __name__ == '__main__':
    dataset = WeishiDetection(image_xml_path='/cephfs/share/data/weishi_xh/train_58_0713.txt', label_file_path='/cephfs/share/data/weishi_xh/label58.txt', transform = None)
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        print("img_id: ", img_id)
