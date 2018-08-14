"""WEISHI Dataset Classes

Updated by: xuhuahuang as intern in YouTu 07/2018
"""
from .config import HOME
import pickle
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import numpy as np
from .weishi_eval import weishi_eval

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# This list will be updated once WeishiAnnotationTransform() is called
WEISHI_CLASSES = ( '__background__',# always index 0
  'null', 'face', 'clothes', 'trousers', 'bag', 'shoes', 'glasses', 'dog', 'cat', 'fish',
  'monkey', 'rabbit', 'bird', 'lobster', 'dolphin', 'panda', 'sheep', 'tiger', 'penguin',
  'turtle', 'lizard', 'snake', 'elephant', 'parrot', 'hamster', 'marmot', 'horse', 'hedgehog',
  'squirrel', 'chicken', 'guitar', 'piano', 'cello_violin', 'saxophone', 'guzheng', 'drum_kit',
  'electronic_organ', 'pipa', 'erhu', 'bike', 'car', 'airplane', 'motorcycle', 'strawberry',
  'banana', 'lemon', 'pig_peggy', 'dead_fish', 'pikachu', 'iron_man', 'spider_man',
  'cell_phone', 'cake', 'cup', 'fountain', 'balloon', 'billards')

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

    def __init__(self, label_file_path = None, class_to_ind=None, keep_difficult=False):
        # change WEISHI_CLASSES if necessary
        if label_file_path is not None:
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

    def __init__(self, root,
                 image_xml_path="input (jpg, xml) file lists",
                 label_file_path = None,
                 transform=None,
                 dataset_name='WEISHI'):
        target_transform=WeishiAnnotationTransform(label_file_path)
        self.root = root # used to store detection results
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = {}
        self._imgpath = {}
        self.name = dataset_name
        # below two args are for evaluation dataset
        self.image_xml_path = image_xml_path
        self.ids = list() # store the names for each image, not useful in WEISHI dataset
        fin = open(image_xml_path, "r")
        count = 0
        for line in fin.readlines():
            line = line.strip()
            des = line.split(' ')
            self._annopath[count] = des[1]
            self._imgpath[count] = des[0]
            count = count + 1
            tree = ET.parse(des[1]).getroot()
            for fname in tree.findall('filename'):
                self.ids.append(fname.text)
                break # only one name
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
        return self.ids[index], gt

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

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        # write down the detection results
        self._write_weishi_results_file(all_boxes)
        # after getting the result file, do evaluation and store in output_dir
        aps, map = self._do_python_eval(output_dir)
        return aps, map

    def _get_weishi_results_file_template(self):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'weishi_det_test' + '_{:s}.txt'
        filedir = os.path.join(self.root, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_weishi_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(WEISHI_CLASSES):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            print('Writing {} WEISHI results file'.format(cls))
            filename = self._get_weishi_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        # for a class in an image: {image_id} {score} {xcor} {xcor} {ycor} {ycor}
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        rootpath = self.root
        name = self.image_set[0][1]
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []
        # Similar to VOC
        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(WEISHI_CLASSES):

            if cls == '__background__':
                continue

            filename = self._get_weishi_results_file_template().format(cls)
            # self is dataset
            rec, prec, ap = weishi_eval(filename, self, \
                                        cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            # AP = AVG(Precision for each of 11 Recalls's precision)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        # MAP = AVG(AP for each object class)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        return aps, np.mean(aps)
