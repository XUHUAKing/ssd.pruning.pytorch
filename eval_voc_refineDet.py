from __future__ import print_function
"""
    Model evaluation on VOC for refineDet separately
    Execute: python3 eval_voc_refineDet.py --trained_model weights/_your_trained_refineDet_model_.pth
    (Take care of different versions of .pth file, can be solved by changing state_dict)
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_CLASSES as labelmap

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

# for evaluation on refineDet
from data import * # val_dataset_root, dataset_root
from layers.box_utils import refine_nms # for detection in test_net for RefineDet
from layers.functions import RefineDetect, PriorBox
from models.RefineSSD_vgg import build_refine
import torch.utils.data as data

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=300, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'
cfg = voc320

priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)
# detector used in test_net for testing
detector = RefineDetect(cfg['num_classes'], 0, cfg, object_score=0.01)

# test function for RefineDet
"""
    Args:
        save_folder: the eval results saving folder
        net: test-type ssd net
        testset: validation dataset
        transform: BaseTransform -- required for refineDet testing,
                   because it pull_image instead of pull_item (this will transform for you)
        max_per_image/top_k： The Maximum number of box preds to consider
"""
def test_net(save_folder, net, detector, priors, cuda,
             testset, transform, max_per_image=300, thresh=0.05): # max_per_image is same as top_k

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    num_classes = len(labelmap)                      # +1 for background
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    #file storing output result under output_dir
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        img = testset.pull_image(i)
        im, _a, _b = transform(img) # to use our incomplete BaseTransform
        im = im.transpose((2, 0, 1))# convert rgb, as extension for our incomplete BaseTransform
        x = Variable(torch.from_numpy(im).unsqueeze(0),volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        arm_loc, arm_conf, odm_loc, odm_conf = out
        boxes, scores = detector.forward((odm_loc,odm_conf), priors, (arm_loc,arm_conf))
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in range(1, num_classes): # for every class
            # for particular class, keep those boxes with score greater than threshold
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds] #filter by inds
            c_scores = scores[inds, j] #filter by inds
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # nms
            keep = refine_nms(c_dets, 0.45) #0.45 is nms threshold
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets #[class][imageID] = 1 x 5 where 5 is box_coord + score

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            # to keep only max_per_image results
            if len(image_scores) > max_per_image:
                # get the smallest score for each class for each image if want to keep only max_per_image results
                image_thresh = np.sort(image_scores)[-max_per_image] # only keep top_k results
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()
        if (i + 1) % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))

    #write the detection results into det_file
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    APs,mAP = testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':
    # load net
    num_classes = len(labelmap)                      # +1 for background
    net = build_refine('test', 320, num_classes, use_refine = True, use_tcb = True) # use_tcb = False
    # if you want to eval refineDet from original version
    '''
    net = build_refine('test', 320, num_classes, use_refine = True, use_tcb = True)
    # load resume network
    resume_net_path = args.trained_model
    print('Loading resume network', resume_net_path)
    state_dict = torch.load(resume_net_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.` because you store the model without DataParallel
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    '''
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(320, dataset_mean),
                           VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, detector, priors, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean),
             args.top_k, thresh=args.confidence_threshold) # 320 originally for cfg['min_dim']
