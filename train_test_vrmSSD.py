'''
    This file support Train + Test SSD model with vgg/resnet/mobilev1/mobilev2 backbone on VOC/XL/WEISHI/COCO dataset

    (Use VOC dataset by default)
    Train + Test SSD model with vgg backbone
    Execute: python3 train_test_vrmSSD.py --evaluate True (testing while training)
    Execute: python3 train_test_vrmSSD.py (only training)

    Train + Test SSD model with resnet backbone
    Execute: python3 train_test_vrmSSD.py --use_res --evaluate True (testing while training)
    Execute: python3 train_test_vrmSSD.py --use_res (only training)

    Train + Test SSD model with mobilev1 backbone
    Execute: python3 train_test_vrmSSD.py --use_m1 --evaluate True (testing while training)
    Execute: python3 train_test_vrmSSD.py --use_m1 (only training)

    Train + Test SSD model with mobilev2 backbone
    Execute: python3 train_test_vrmSSD.py --use_m2 --evaluate True (testing while training)
    Execute: python3 train_test_vrmSSD.py --use_m2 (only training)

    (Use WEISHI dataset)
    --dataset WEISHI --dataset_root _path_for_WEISHI_ROOT --jpg_xml_path _path_of_your_jpg_xml

    (Use XL dataset)
    --dataset XL --dataset_root _path_for_XL_ROOT

    Author: xuhuahuang as intern in YouTu 07/2018
'''
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models.SSD_vggres import build_ssd
from models.SSD_mobile import build_mssd
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
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'WEISHI', 'XL'],
                    type=str, help='VOC or COCO or WEISHI or XL') #'XL', for VOC_xlab_products
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path') #XL_ROOT, for VOC_xlab_products
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
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
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
#parser.add_argument('--top_k', default=5, type=int,
#                    help='Further restrict the number of predictions to parse')
# for WEISHI dataset
parser.add_argument('--jpg_xml_path', default='', #'/cephfs/share/data/weishi_xh/train_58_0522.txt'
                    help='Image XML mapping path')
parser.add_argument('--label_name_path', default=None, #'/cephfs/share/data/weishi_xh/label58.txt'
                    help='Label Name file path')
# for resnet backbone
parser.add_argument("--use_res", dest="use_res", action="store_true")
parser.set_defaults(use_res=False)
# for mobilev1 backbone
parser.add_argument("--use_m1", dest="use_m1", action="store_true")
parser.set_defaults(use_m1=False)
# for mobilev2 backbone
parser.add_argument("--use_m2", dest="use_m2", action="store_true")
parser.set_defaults(use_m2=False)

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

# train/val dataset set-up
if args.dataset == 'VOC':
    if args.dataset_root == COCO_ROOT:
        parser.error('Must specify dataset if specifying dataset_root')
    cfg = voc
    dataset = VOCDetection(root=args.dataset_root, \
                           transform=SSDAugmentation(cfg['min_dim'], cfg['dataset_mean']))
    val_dataset = VOCDetection(root=voc_val_dataset_root, image_sets=[('2007', 'test')], \
                               transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) # 300 originally
elif args.dataset == 'XL':
    if args.dataset_root != XL_ROOT:
        parser.error('Must specify dataset_root if using XL')
    cfg = xl
    dataset = XLDetection(root=args.dataset_root, \
                          transform=SSDAugmentation(cfg['min_dim'], cfg['dataset_mean']))
    val_dataset = XLDetection(root=xl_val_dataset_root, image_sets=['test'], \
                              transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) # 300 originally
elif args.dataset == 'WEISHI':
    if args.jpg_xml_path == '':
        parser.error('Must specify jpg_xml_path if using WEISHI')
    cfg = weishi
    dataset = WeishiDetection(root=args.dataset_root, \
                              image_xml_path=args.jpg_xml_path, label_file_path=args.label_name_path, \
                              transform=SSDAugmentation(cfg['min_dim'], cfg['dataset_mean']))
    val_dataset = WeishiDetection(root = weishi_val_dataset_root, \
                                  image_xml_path=weishi_val_imgxml_path, label_file_path=args.label_name_path, \
                                  transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) # 300 originally
elif args.dataset == 'COCO':
    if args.dataset_root == VOC_ROOT:
        if not os.path.exists(COCO_ROOT):
            parser.error('Must specify dataset_root if specifying dataset')
        print("WARNING: Using default COCO dataset_root because " +
              "--dataset_root was not specified.")
        args.dataset_root = COCO_ROOT
    cfg = coco
    # TODO: evaluation on COCO dataset
    dataset = COCODetection(root=args.dataset_root, \
                            transform=SSDAugmentation(cfg['min_dim'], cfg['dataset_mean']))
    val_dataset = COCODetection(root=coco_val_dataset_root, \
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean'])) # 300 originally

def train():
    # network set-up
    if args.use_res:
        ssd_net = build_ssd('train', cfg, cfg['min_dim'], cfg['num_classes'], base='resnet') # for resnet
    elif args.use_m1:
        ssd_net = build_mssd('train', cfg, cfg['min_dim'], cfg['num_classes'], base='m1') # backbone network is m1
    elif args.use_m2:
        ssd_net = build_mssd('train', cfg, cfg['min_dim'], cfg['num_classes'], base='m2') # backbone network is m2
    else:
        ssd_net = build_ssd('train', cfg, cfg['min_dim'], cfg['num_classes'], base='vgg') # backbone network is vgg
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        print('Using preloaded base network...') # Preloaded.
        print('Initializing other weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if args.cuda:
        net = net.cuda()

    # training set-up
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
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
    for iteration in range(args.start_iter, cfg['max_epoch']*epoch_size + 10):
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)# the dataloader cannot re-initilize
            images, targets = next(batch_iterator)

        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration != 0 and (iteration % epoch_size == 0):
            # adjust_learning_rate(optimizer, args.gamma, epoch)
            # evaluation during training
            if args.evaluate == True:
                # load net
                net.eval()
                top_k = (300, 200)[args.dataset == 'COCO'] # for VOC_xlab_products
                APs,mAP = test_net(args.eval_folder, net, args.cuda, val_dataset,
                         BaseTransform(net.module.size, cfg['testset_mean']),
                         top_k, thresh=args.confidence_threshold) # 300 is for cfg['min_dim'] originally
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
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Total iter ' +
                  repr(iteration) + ' || Loc: %.4f Conf: %.4f||' % (loss_l, loss_c) +
                'Timer: %.4f sec. ||' % (t1 - t0) + 'Loss: %.4f ||' % (loss.data[0]) + 'LR: %.8f' % (lr))

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            if args.use_res:
                torch.save(ssd_net.state_dict(), 'weights/ssd300_resnet_' + # for resnet
                           repr(iteration) + '.pth')
            elif args.use_m1:
                torch.save(ssd_net.state_dict(), 'weights/ssd300_mobilev1_' +
                           repr(iteration) + '.pth')
            elif args.use_m2:
                torch.save(ssd_net.state_dict(), 'weights/ssd300_mobilev2_' +
                           repr(iteration) + '.pth')
            else:
                torch.save(ssd_net.state_dict(), 'weights/ssd300_vgg_' +
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

def xavier(param):
    init.xavier_uniform(param)

# initialize the weights for conv2d
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

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

# test function for SSD
"""
    Args:
        save_folder: the eval results saving folder
        net: test-type ssd net
        dataset: validation dataset
        transform: BaseTransform - not used here
        max_per_image/top_k：The Maximum number of box preds to consider
"""
def test_net(save_folder, net, cuda,
             testset, transform, max_per_image=300, thresh=0.05):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    num_classes = cfg['num_classes']
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    #output_dir = get_output_dir('ssd300_120000', set_type) #directory storing output results
    #det_file = os.path.join(output_dir, 'detections.pkl') #file storing output result under output_dir
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = testset.pull_item(i) # include BaseTransform inside

        x = Variable(im.unsqueeze(0)) #insert a dimension of size one at the dim 0
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        detections = net(x=x, test=True).data # get the detection results
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

        if (i + 1) % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time)) # nms time is included in detect_time for normal SSD

    #write the detection results into det_file
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    APs,mAP = testset.evaluate_detections(all_boxes, save_folder)
    return APs,mAP

if __name__ == '__main__':
    train()
