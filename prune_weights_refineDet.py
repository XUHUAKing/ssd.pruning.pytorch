'''
    Use absolute weights-based criterion for filter pruning on refineDet(vgg)
    Execute: python3 prune_weights_refineDet.py --trained_model weights/_your_trained_model_.pth
    Author: xuhuahuang as intern in YouTu 07/2018
'''
import torch
from torch.autograd import Variable
#from torchvision import models
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import dataset
from pruning.prune_tools import *
import argparse
from operator import itemgetter
from heapq import nsmallest #heap queue algorithm
import time

# for testing
import pickle
import os
from data import * # BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from layers.box_utils import refine_nms # for detection in test_net for RefineDet
from layers.modules import RefineMultiBoxLoss
from layers.functions import RefineDetect, PriorBox
from models.RefineSSD_vgg import build_refine

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--prune_folder", default = "prunes/")
parser.add_argument("--trained_model", default = "prunes/refineDet_trained.pth")
parser.add_argument('--dataset_root', default=VOC_ROOT)
parser.add_argument("--cut_ratio", default=0.2, type=int)
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
args = parser.parse_args()

cfg = voc320

# different from normal ssd, where the PriorBox is stored inside SSD object
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward().cuda(), volatile=True) # set the priors to cuda
detector = RefineDetect(cfg['num_classes'], 0, cfg, object_score=0.01)

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

# --------------------------------------------------------------------------- Pruning Part
class Prunner_refineDet:
    def __init__(self, testset, arm_criterion, odm_criterion, model):
        self.testset = testset

        self.model = model
        self.arm_criterion = arm_criterion
        self.odm_criterion = odm_criterion
        self.model.train()

    def test(self):
        self.model.eval()
        # evaluation
        test_net('prunes/test', self.model, detector, priors, args.cuda, testset,
                 BaseTransform(self.model.size, cfg['dataset_mean']),
                 300, thresh=0.01)

        self.model.train()

    def prune(self, cut_ratio = 0.2):
        #Get the accuracy before prunning
        self.test()

        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.base.parameters():
            param.requires_grad = True

        fork_indices = [21, 28, 33] #len(self.model.base)-1 = 34
        for layer, (name, module) in enumerate(self.model.base._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d) and (layer not in fork_indices):

                print("Pruning layer ", layer, "..")
                model = self.model.cpu()
                model = prune_conv_layer(model, layer, cut_ratio=cut_ratio, use_bn = False)
                self.model = model.cuda()
                # self.test()

        print("Finished. Get the accuracy after pruning..")
        self.test()

        print('Saving pruned model...')
        torch.save(self.model, 'prunes/refineDet_prunned')

if __name__ == '__main__':
    if not args.cuda:
        print("this file only supports cuda version now!")

    # store pruning models
    if not os.path.exists(args.prune_folder):
        os.mkdir(args.prune_folder)

    # ------------------------------------------- 1st prune: load model from state_dict
    model = build_refine('train', cfg['min_dim'], cfg['num_classes'], use_refine = True, use_tcb = True).cuda()
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7] # head = k[:4]
        if head == 'module.': # head == 'vgg.'
            name = k[7:]  # name = 'base.' + k[4:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(torch.load(args.trained_model))
    # ------------------------------------------- >= 2nd prune: load model from previous pruning
    # model = torch.load(args.trained_model).cuda()

    print('Finished loading model!')

    testset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'test')],
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean']))
    arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, 0, args.cuda)
    odm_criterion = RefineMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, 0.01, args.cuda)# 0.01 -> 0.99 negative confidence threshold

    prunner = Prunner_refineDet(testset, arm_criterion, odm_criterion, model)
    prunner.prune(cut_ratio = args.cut_ratio)
