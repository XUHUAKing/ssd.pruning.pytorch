"""WEISHI Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# This list will be updated once WeishiAnnotationTransform() is called
# fake classes temporary
WEISHI_CLASSES = ( #always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class WeishiAnnotationTransform(object):
    """Transforms a Weishi annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, label_file_path = "", class_to_ind=None, keep_difficult=False):
        global WEISHI_CLASSES # declare that WEISHI_CLASSES is changed globally by this function
        WEISHI_CLASSES = list()
        fin = open(label_file_path, 'r')
        for line in fin.readlines():
            line = line.strip()
            WEISHI_CLASSES.append(line)
        fin.close()
        WEISHI_CLASSES = tuple(WEISHI_CLASSES)
        self.class_to_ind = class_to_ind or dict(
            zip(WEISHI_CLASSES, range(len(WEISHI_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element.
            target has been ET.Element type already when being passed inside
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class WeishiDetection(data.Dataset):
    """Weishi Detection Dataset Object

    input is image, target is annotation

    Arguments:
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        _annopath: path for a specific annotation, extract from .txt file later
        _imgpath: path for a specific image, extract from .txt file later
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, image_xml_path="input (jpg, xml) file lists", label_file_path = "",
                 transform=None):
        target_transform=WeishiAnnotationTransform(label_file_path)
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = {}
        self._imgpath = {}
        self.name = "weishi"
        # below two args are for evaluation dataset
        self.image_xml_path = image_xml_path
        self.ids = list() # store the names for each image
        fin = open(image_xml_path, "r")
        count = 0
        for line in fin.readlines():
            line = line.strip()
            des = line.split(' ')
            self._annopath[count] = des[1]
            tree = ET.parse(des[1])
            for fname in tree.findall('filename'):
                self.ids.append(fname.text)
                break # only one name
            self._imgpath[count] = des[0]
            count = count + 1
        fin.close()

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self._imgpath)

    def pull_item(self, index):
        target = ET.parse(self._annopath[index]).getroot()
        img = cv2.imread(self._imgpath[index])
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        return cv2.imread(self._imgpath[index], cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        anno = ET.parse(self._annopath[index]).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
