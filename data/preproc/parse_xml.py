"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import sys
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

#target = ET.parse("/cephfs/group/youtu/data/OpenSourceDetData/openimage/xml0508/train/4/446ab75d5c9e48be.xml").getroot()
target = ET.parse("/cephfs/group/youtu/data/OpenSourceDetData/openimage/xml0508/train/4/446ab75d5c9e48be.xml").getroot()
target = np.array(target)
print(target.shape)
print(target)
