'''
    Train + Test refineDet model
    Execute: python3 train_test_refineDet.py --evaluate True (testing while training)
    Execute: python3 train_test_refineDet.py (only training)
'''
from data import * # val_dataset_root, dataset_root, Timer
from data import VOC_CLASSES as voc_labelmap
from data import COCO_CLASSES as coco_labelmap
from data import WEISHI_CLASSES as weishi_labelmap
from utils.augmentations import SSDAugmentation
from layers.box_utils import refine_nms # for detection in test_net for RefineDet
from layers.modules import RefineMultiBoxLoss
from layers.functions import RefineDetect, PriorBox
from models.RefineSSD_vgg import build_refine
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pickle

#os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Refinement SSD Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'WEISHI'],
                    type=str, help='VOC or COCO or WEISHI')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('-we','--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_step', default=30,
                    help='Epoch interval for updating lr')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
# Below args must be specified if want to eval during training
parser.add_argument('--evaluate', default=False, type=str2bool,
                    help='Evaluate at every epoch during training')
parser.add_argument('--eval_folder', default='evals/',
                    help='Directory for saving eval results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
# top_k = (300, 200)[args.dataset == 'COCO']
#parser.add_argument('--top_k', default=5, type=int,
#                    help='Further restrict the number of predictions to parse')
# for WEISHI dataset
parser.add_argument('--jpg_xml_path', default='',
                    help='Image XML mapping path')
parser.add_argument('--label_name_path', default='',
                    help='Label Name file path')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.eval_folder):
    os.mkdir(args.eval_folder)


def train():
    # train/val dataset object initialization
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          cfg['dataset_mean']))
        val_dataset = COCODetection(root=coco_val_dataset_root,
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) #320 originally
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc320 # min_dim inside will ask SSDAugmentation change size of picture
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         cfg['dataset_mean']))
        val_dataset = VOCDetection(root=voc_val_dataset_root, image_sets=[('2007', 'test')],
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) # 320 originally
    elif args.dataset == 'WEISHI':
        if args.jpg_xml_path == '':
            parser.error('Must specify jpg_xml_path if using WEISHI')
        if args.label_name_path == '':
            parser.error('Must specify label_name_path if using WEISHI')
        cfg = weishi
        dataset = WeishiDetection(image_xml_path=args.jpg_xml_path, label_file_path=args.label_name_path,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         cfg['dataset_mean']))
        val_dataset = WeishiDetection(image_xml_path=weishi_val_imgxml_path, label_file_path=args.label_name_path,
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) # 320 originally

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    # network set-up
    ssd_net = build_refine('train', cfg['min_dim'], cfg['num_classes'], use_refine = True, use_tcb = True)
    net = ssd_net
    print(net)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net) # state_dict will have .module. prefix
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.base.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.trans_layers.apply(weights_init)
        ssd_net.latent_layrs.apply(weights_init)
        ssd_net.up_layers.apply(weights_init)
        ssd_net.arm_loc.apply(weights_init)
        ssd_net.arm_conf.apply(weights_init)
        ssd_net.odm_loc.apply(weights_init)
        ssd_net.odm_conf.apply(weights_init)

    # otimizer and loss set-up
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, 0, args.cuda)
    odm_criterion = RefineMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, 0.01, args.cuda)# 0.01 -> 0.99 negative confidence threshold

    # different from normal ssd, where the PriorBox is stored inside SSD object
    priorbox = PriorBox(cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    # detector used in test_net for testing
    detector = RefineDetect(cfg['num_classes'], 0, cfg, object_score=0.01)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training refineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    if args.visdom:
        # initialize visdom loss plot
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    # adjust learning rate based on epoch
    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    step_index = 0

    # training data loader
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
#    batch_iterator = None
    mean_odm_loss_c = 0
    mean_odm_loss_l = 0
    mean_arm_loss_c = 0
    mean_arm_loss_l = 0
    # max_iter = cfg['max_epoch'] * epoch_size
    for iteration in range(args.start_iter, cfg['max_epoch']*epoch_size + 10):
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)# the dataloader cannot re-initilize
            images, targets = next(batch_iterator)
        '''
        if (iteration % epoch_size == 0):
            # create/update batch iterator at every epoch - re-initilize
            batch_iterator = iter(data.DataLoader(dataset, args.batch_size,
                                                  num_workers=args.num_workers,
                                                  shuffle=True, collate_fn=detection_collate))
        # load train data
        images, targets = next(batch_iterator)
        '''

        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            # update visdom loss plot
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration != 0 and (iteration % epoch_size == 0):
    #        adjust_learning_rate(optimizer, args.gamma, epoch)
            # evaluation
            if args.evaluate == True:
                # load net
                net.eval()
                top_k = (300, 200)[args.dataset == 'COCO']
                if args.dataset == 'VOC':
                    APs,mAP = test_net(args.eval_folder, net, detector, priors, args.cuda, val_dataset,
                             BaseTransform(net.module.size, cfg['testset_mean']),
                             top_k, thresh=args.confidence_threshold) # 320 originally for cfg['min_dim']
                else:#COCO
                    test_net(args.eval_folder, net, detector, priors, args.cuda, val_dataset,
                             BaseTransform(net.module.size, cfg['testset_mean']),
                             top_k, thresh=args.confidence_threshold) # DataParallel object should have module for net.module.size

                net.train()
            epoch += 1

        # update learning rate
        if iteration in stepvalues:
            step_index  = stepvalues.index(iteration) + 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        arm_loc, arm_conf, odm_loc, odm_conf = out
        # backprop
        optimizer.zero_grad()
        #arm branch loss
        #priors = priors.type(type(images.data)) #convert to same datatype
        arm_loss_l,arm_loss_c = arm_criterion((arm_loc,arm_conf),priors,targets)
        #odm branch loss
        odm_loss_l, odm_loss_c = odm_criterion((odm_loc,odm_conf),priors,targets,(arm_loc,arm_conf),False)

        mean_arm_loss_c += arm_loss_c.data[0]
        mean_arm_loss_l += arm_loss_l.data[0]
        mean_odm_loss_c += odm_loss_c.data[0]
        mean_odm_loss_l += odm_loss_l.data[0]

        loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Total iter ' +
                  repr(iteration) + ' || AL: %.4f AC: %.4f OL: %.4f OC: %.4f||' % (
                mean_arm_loss_l/10,mean_arm_loss_c/10,mean_odm_loss_l/10,mean_odm_loss_c/10) +
                'Timer: %.4f sec. ||' % (t1 - t0) + 'Loss: %.4f ||' % (loss.data[0]) + 'LR: %.8f' % (lr))

            mean_odm_loss_c = 0
            mean_odm_loss_l = 0
            mean_arm_loss_c = 0
            mean_arm_loss_l = 0

#        if args.visdom:
#            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
#                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_refineDet_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * args.warm_epoch)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
'''
def adjust_learning_rate(optimizer, gamma, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at
        specified epoch
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    step = epoch//args.lr_step # every 30 epoch by default
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''

def xavier(param):
    init.xavier_uniform(param)

# initialize the weights for conv2d
def weights_init(m):
    '''
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    '''
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

# test function for RefineDet
"""
    Args:
        save_folder: the eval results saving folder
        net: test-type ssd net
        testset: validation dataset
        transform: BaseTransform -- required for refineDet testing,
                   because it pull_image instead of pull_item (this will transform for you)
"""
def test_net(save_folder, net, detector, priors, cuda,
             testset, transform, max_per_image=300, thresh=0.05): # max_per_image is same as top_k

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
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
            # keep, _ = nms(torch.from_numpy(c_bboxes), torch.from_numpy(c_scores), 0.45, top_k) #0.45 is nms threshold
            keep = refine_nms(c_dets, 0.45) #0.45 is nms threshold
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets #[class][imageID] = 1 x 5 where 5 is box_coord + score

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            # to keep only max_per_image results
            if len(image_scores) > max_per_image:
                # get the smallest score for each class for each image if want to keep only max_per_image results
                image_thresh = np.sort(image_scores)[-max_per_image] # top_k
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
    if args.dataset == 'VOC':
        APs,mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs,mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':
    train()
