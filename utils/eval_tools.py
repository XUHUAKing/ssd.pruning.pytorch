"""
    This file includes util tools for evaluation during training
    Tools for mAP evaluation
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as voc_labelmap
from data import COCO_CLASSES as coco_labelmap
from data import WEISHI_CLASSES as weishi_labelmap
import torch.utils.data as data

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


"""FOR VOC"""
# the val dataset root
voc_val_dataset_root = '/cephfs/share/data/VOCdevkit/'

annopath = os.path.join(voc_val_dataset_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(voc_val_dataset_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(voc_val_dataset_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = voc_val_dataset_root + 'VOC' + YEAR
voc_dataset_mean = (104, 117, 123) #val dataset mean

"""FOR COCO"""
# the val dataset root
coco_val_dataset_root = '/cephfs/share/data/coco_xy/'
coco_path = coco_val_dataset_root
coco_dataset_mean = (104, 117, 123) #val dataset mean

"""FOR WEISHI"""
weishi_val_dataset_root = ''
weishi_val_imgxml_path = ''
weishi_path = weishi_val_dataset_root
weishi_dataset_mean = (104, 117, 123) #val dataset mean

# global variable
val_dataset_root = voc_val_dataset_root #default
dataset_path = devkit_path
dataset_mean = (104, 117, 123)
labelmap = voc_labelmap
gset = 'voc' # dataset type

set_type = 'test' # for every one


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    """ Parse a WEISHI xml file """
    tree = ET.parse(filename) #filename is the path
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


#def get_voc_results_file_template(image_set, cls):
def get_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(dataset_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


#def write_voc_results_file(all_boxes, dataset, labelmap):
def write_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        #print('Writing {:s} VOC results file'.format(cls))
        print('Writing {:s} results file'.format(cls))
        #filename = get_voc_results_file_template(set_type, cls)
        filename = get_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]# add index 1 for background class
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices ????
                for k in range(dets.shape[0]):# for a class in an image
                    # {image_id} {score} {xcor} {xcor} {ycor} {ycor}
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(dataset, output_dir='output', use_07=True):
    cachedir = os.path.join(dataset_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        #filename = get_voc_results_file_template(set_type, cls)
        filename = get_results_file_template(set_type, cls)
        if gset == 'weishi':
            rec, prec, ap = weishi_eval(
                filename, dataset, cls, cachedir,
                ovthresh=0.5, use_07_metric=use_07_metric)
        elif gset == 'coco':
            rec, prec, ap = coco_eval()
        else: # voc by default
            rec, prec, ap = voc_eval(
               filename, annopath, imgsetpath.format(set_type), cls, cachedir,
               ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap] # AP = AVG(Precision for each of 11 Recalls's precision)
        print('AP for {} = {:.4f}'.format(cls, ap))
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

#def voc_ap(rec, prec, use_07_metric=true):
def ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1): # [0, 0.1, 0.2, 0.3 ..., 1]
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

"""FOR VOC"""
"""
rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching/storing the annotations .pkl
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# for a specific class
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    # recs stores the annots for each images
    # class_recs stores the gt for a class

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # go through every image
    for imagename in imagenames:
        # and extract those objects in this image that are under this designated class
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R]) # the object belongs to this class
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # for each image, store the bboxs for this class inside this image
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines] # [[image_id1, confidence1, xmin1, xmax1, ymin1, ymax1], [], [], []...]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)# find the vertex of intersected rectangle
                ih = np.maximum(iymax - iymin, 0.)# find the vertex of intersected rectangle
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:# ignore difficult
                    if not R['det'][jmax]: #R['det'][jmax] has NOT already been 1
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1. # false positive
            else:
                fp[d] = 1. #false positive

        # compute precision recall
        fp = np.cumsum(fp)# how many 1 in fp array
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

"""FOR COCO"""
def coco_eval():
    # use the official COCO evaluation code
    pass

"""FOR WEISHI"""
def weishi_eval(detpath,
                dataset,
                classname,
                cachedir,
                ovthresh=0.5,
                use_07_metric=True):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    imagenames = dataset.ids# a list of image ids
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        # read list of images
        fin = open(dataset.image_xml_path, 'r')
        for i, line in enumerate(fin.readlines()):
            line = line.strip()
            des = line.split(' ')
            annopath = des[1]
            imagename = imagenames[i]
            recs[imagename] = parse_rec(annopath)
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    # recs stores the annots for each images
    # class_recs stores the gt for a class

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # go through every image
    for imagename in imagenames:
        # and extract those objects in this image that are under this designated class
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R]) # the object belongs to this class
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # for each image, store the bboxs for this class inside this image
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines] # [[image_id1, confidence1, xmin1, xmax1, ymin1, ymax1], [], [], []...]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)# find the vertex of intersected rectangle
                ih = np.maximum(iymax - iymin, 0.)# find the vertex of intersected rectangle
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:# ignore difficult
                    if not R['det'][jmax]: #R['det'][jmax] has NOT already been 1
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1. # false positive
            else:
                fp[d] = 1. #false positive

        # compute precision recall
        fp = np.cumsum(fp)# how many 1 in fp array
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

"""
    Args:
        save_folder: the eval results saving folder
        net: test-type ssd net
        dataset: validation dataset
        transform: BaseTransform
        labelmap: labelmap for different dataset (voc, coco, weishi)
"""
def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05, set='voc'):
    global val_dataset_root, dataset_path, dataset_mean, labelmap, gset
    # update global variable
    gset = set
    if set == 'voc':
        val_dataset_root = voc_val_dataset_root
        dataset_path = devkit_path
        dataset_mean = voc_dataset_mean
        labelmap = voc_labelmap
    elif set == 'coco':
        val_dataset_root = coco_val_dataset_root
        dataset_path = coco_path
        dataset_mean = coco_dataset_mean
        labelmap = coco_labelmap
    elif set == 'weishi':
        val_dataset_root = weishi_val_dataset_root
        dataset_path = weishi_path
        dataset_mean = weishi_dataset_mean
        labelmap = weishi_labelmap

    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type) #directory storing output results
    det_file = os.path.join(output_dir, 'detections.pkl') #file storing output result under output_dir

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0)) #insert a dimension of size one at the dim 0
        if cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data # get the detection results
        detect_time = _t['im_detect'].toc(average=False) #store the detection time

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)): # for every class
            dets = detections[0, j, :]#size( ** , 5)
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            #if dets.size(0) == 0:
            #    continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets #[class][imageID] = 1 x 5 where 5 is box_coord + score

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:#write the detection results into det_file
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)

def evaluate_detections(box_list, output_dir, dataset):
    #write_voc_results_file(box_list, dataset, labelmap)
    write_results_file(box_list, dataset) # write down the detetcion results
    do_python_eval(output_dir=output_dir, dataset=dataset) # after getting the result file, do evaluation and store in output_dir
