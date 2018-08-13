'''
    Finetune prunned model refineDet(vgg) (Train/Test on VOC)
    Execute: python3 finetune_refineDet.py --pruned_model prunes/_your_prunned_model --lr x --epoch y
    
    Author: xuhuahuang as intern in YouTu 07/2018
    Status: checked
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
import argparse
from operator import itemgetter
import time

# for testing
import pickle
import os
from data import * # BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from utils.augmentations import SSDAugmentation
from layers.box_utils import refine_nms # for detection in test_net for RefineDet
from layers.modules import RefineMultiBoxLoss
from layers.functions import RefineDetect, PriorBox
from models.RefineSSD_vgg import build_refine

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--prune_folder", default = "prunes/")
parser.add_argument("--pruned_model", default = "prunes/refineDet_prunned")
parser.add_argument('--dataset_root', default=VOC_ROOT)
parser.add_argument("--cut_ratio", default=0.2, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--epoch", default=20, type=int)
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

    return mAP # for model storing

# --------------------------------------------------------------------------- Finetune Part
class FineTuner_refineDet:
    def __init__(self, train_loader, testset, arm_criterion, odm_criterion, model):
        self.train_data_loader = train_loader
        self.testset = testset

        self.model = model
        self.arm_criterion = arm_criterion
        self.odm_criterion = odm_criterion
        self.model.train()

    def test(self):
        self.model.eval()
        # evaluation
        map = test_net('prunes/test', self.model, detector, priors, args.cuda, testset,
                 BaseTransform(self.model.size, cfg['dataset_mean']),
                 300, thresh=0.01)

        self.model.train()
        return map

    # epoches: fine tuning for this epoches
    def train(self, optimizer = None, epoches = 5):
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.parameters(),
                    lr=0.0001, momentum=0.9, weight_decay=5e-4)

        for i in range(epoches):
            print("FineTune... Epoch: ", i+1)
            self.train_epoch(optimizer) # no need for rank filters
            map = self.test()
        print("Finished fine tuning. mAP is ", map)
        return map

    # batch: images, label: targets
    def train_batch(self, optimizer, batch, label):
        # set gradients of all model parameters to zero
        self.model.zero_grad() # same as optimizer.zero_grad() when SGD() get model.parameters
        # input = Variable(batch)
        input = batch

        arm_loc, arm_conf, odm_loc, odm_conf = self.model(input)
        arm_loss_l, arm_loss_c = self.arm_criterion((arm_loc, arm_conf), priors, label)#Variable(label))
        odm_loss_l, odm_loss_c = self.odm_criterion((odm_loc, odm_conf), priors, label,(arm_loc,arm_conf), False)
        loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step() # update params

    # train for one epoch, so the data_loader will not pop StopIteration error
    def train_epoch(self, optimizer = None):
        num_batch = 0
        for batch, label in self.train_data_loader:
            num_batch += 1
            if num_batch % 50 == 0:
                print("Training batch " + repr(num_batch) + "/" + repr(len(self.train_data_loader)-1) + "...")
            batch = Variable(batch.cuda())
            label = [Variable(ann.cuda(), volatile=True) for ann in label]
            self.train_batch(optimizer, batch, label)

if __name__ == '__main__':
    if not args.cuda:
        print("this file only supports cuda version now!")

    # store pruning models
    if not os.path.exists(args.prune_folder):
        os.mkdir(args.prune_folder)

    print(args)
    # load model from previous pruning
    model = torch.load(args.pruned_model).cuda()
    print('Finished loading model!')

    # data
    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'], cfg['dataset_mean']))
    testset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'test')],
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean']))
    data_loader = data.DataLoader(dataset, 32, num_workers=4,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, 0, args.cuda)
    odm_criterion = RefineMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, 0.01, args.cuda)# 0.01 -> 0.99 negative confidence threshold

    fine_tuner = FineTuner_refineDet(data_loader, testset, arm_criterion, odm_criterion, model)

    # ------------------------ adjustable part
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    map = fine_tuner.train(optimizer = optimizer, epoches = args.epoch)
    # ------------------------ adjustable part

    print('Saving finetuned model with map ', map, '...')
    torch.save(model, 'prunes/refineDet_finetuned_{0:.2f}'.format(map*100)) # same as fine_tuner.model
