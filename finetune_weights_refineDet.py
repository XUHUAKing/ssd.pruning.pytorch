'''
	Use absolute weights-based criterion for filter pruning on refineDet(vgg)
'''
import torch
from torch.autograd import Variable
#from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import dataset
from pruning.prune_vgg import *
import argparse
from operator import itemgetter
from heapq import nsmallest #heap queue algorithm
import time

from data import *
import torch.utils.data as data
from utils.augmentations import SSDAugmentation
from layers.modules import RefineMultiBoxLoss
from layers.functions import RefineDetect, PriorBox
from models.RefineSSD_vgg import build_refine

# only prune base net
# store the functions for ranking and deciding which filters to be pruned
class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.weights = [] # store the absolute weights for filter
        self.weight_to_layer = {} # dict matching weight index to layer index

        weight_index = 0
        # the layer excluded from pruning due to existence of forking
        fork_indices = [21, 28, len(self.model.base)-1]
        for layer, (name, module) in enumerate(self.model.base._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d) and (layer not in fork_indices):
                print(module.weight.data.size()) # batch_size x out_channels x 3 x 3?

                abs_wgt = torch.abs(module.weight.data) # batch_size x out_channels x 3 x 3?
                self.weights.append(abs_wgt)
                self.weight_to_layer[weight_index] = layer
                weight_index += 1

                # compute the rank and store into self.filter_ranks
                # size(1) represents the num of filter/individual feature map
                values = \
                    torch.sum(abs_wgt, dim = 0).\
                        sum(dim=2).sum(dim=3)[0, :, 0, 0].data

                # Normalize the sum of weight by the batch_size
                values = values / abs_wgt.size(0) # (filter_number for this layer, 1)

                if weight_index not in self.filter_ranks:
                    self.filter_ranks[weight_index] = \
                        torch.FloatTensor(abs_wgt.size(1)).zero_().cuda()

                self.filter_ranks[weight_index] += values # filter_ranks are 0 initially, size = (num_filter, 1)

        return self.model(x) # output

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()): # for one layer
            for j in range(self.filter_ranks[i].size(0)): # num_filter for this layer
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2)) # (l, f, _)

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            # After each of the k filters are prunned,
            # the filter index of the next filters change since the model is smaller.
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

class PrunningFineTuner_refineDet:
    def __init__(self, train_loader, arm_criterion, odm_criterion, model):
        self.train_data_loader = train_loader

        self.model = model
        self.arm_criterion = arm_criterion
        self.odm_criterion = odm_criterion
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self, i):
        print("Saving pruned model at epoch/iteration {}...".format(i))
        torch.save(self.model.state_dict(), 'prunes/refineDetmodel_for_'+ repr(i) +'_test.pth')
        print("Model saved, please use eval_.py for evaluating.")

    # epoches: fine tuning for this epoches
    def train(self, priors, optimizer = None, epoches = 10):
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.parameters(),
                    lr=0.0001, momentum=0.9, weight_decay=5e-4)

        for i in range(epoches):
            print("FineTune... Epoch: ", i)
            self.train_epoch(priors, optimizer) # no need for rank_filters
            self.test(i)
        print("Finished fine tuning.")

    # batch: images, label: targets
    def train_batch(self, optimizer, batch, priors, label, rank_filters):
        # set gradients of all model parameters to zero
        self.model.zero_grad() # same as optimizer.zero_grad() when SGD() get model.parameters
        input = Variable(batch)

        # just for ranking the filter, not for params update, use self.prunner for output
        if rank_filters:
			arm_loc, arm_conf, odm_loc, odm_conf = self.prunner.forward(input)
            arm_loss_l, arm_loss_c = self.arm_criterion((arm_loc, arm_conf), priors, Variable(label))
	        odm_loss_l, odm_loss_c = self.odm_criterion((odm_loc, odm_conf), priors, Variable(label),(arm_loc,arm_conf), False)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
            loss.backward()
        else:
			arm_loc, arm_conf, odm_loc, odm_conf = self.model(input)
            arm_loss_l, arm_loss_c = self.arm_criterion((arm_loc, arm_conf), priors, Variable(label))
	        odm_loss_l, odm_loss_c = self.odm_criterion((odm_loc, odm_conf), priors, Variable(label),(arm_loc,arm_conf), False)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
            loss.backward()
            optimizer.step() # update params

    # train for one epoch, so the data_loader will not pop StopIteration error
    def train_epoch(self, priors, optimizer = None, rank_filters = False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch.cuda(), priors, label.cuda(), rank_filters)

    def get_candidates_to_prune(self, priors, num_filters_to_prune):
        self.prunner.reset()

        self.train_epoch(priors, rank_filters = True) # need to rank_filters

        # self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.base._modules.items():
	        fork_indices = [21, 28, len(self.model.base)-1]
            if isinstance(module, torch.nn.modules.conv.Conv2d) and (layer not in fork_indices):
                filters = filters + module.out_channels
        return filters

    def prune(self, priors):
        #Get the accuracy before prunning
        self.test(00)

        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.base.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration) # the total iterations for cutting 100% all

        #iterations = int(iterations * 2.0 / 3)
        iterations = int(iterations * 2.0 / 10)

        #print "Number of prunning iterations to reduce 67% filters", iterations
        print("Number of prunning iterations to reduce 20% filters", iterations)

        for iteration in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vggbase_conv_layer(model, layer_index, filter_index)

            self.model = model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test(iteration)
            print("Fine tuning to recover from prunning iteration.")

		    # otimizer and loss set-up
		    optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

            self.train(priors, optimizer, epoches = 10)

        print("Finished. Going to fine tune the model a bit more")
        self.train(priors, optimizer, epoches = 15)
        print('Saving pruned model...')
        torch.save(self.model.state_dict(), 'prunes/Refinemodel_prunned.pth')
        #torch.save(model, "model_prunned")

def get_args():
    parser = argparse.argumentparser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--prune_folder", default = "prunes/")
    parser.add_argument("--trained_model", default = "prunes/Refinemodel_trained.pth")
    parser.add_argument('--dataset_root', default=voc_root)
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    cfg = voc320
    cuda = True
    # store pruning models
    if not os.path.exists(args.prune_folder):
        os.mkdir(args.prune_folder)

    if args.train:
		model = build_refine('train', cfg['min_dim'], cfg['num_classes'], use_refine = True, use_tcb = True).cuda()
    elif args.prune:
        #model = torch.load("model").cuda()
		model = build_refine('train', cfg['min_dim'], cfg['num_classes'], use_refine = True, use_tcb = True).cuda()
        state_dict = torch.load(args.trained_model)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.` because you store the model without DataParallel
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        #model.load_state_dict(torch.load(args.trained_model))

    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     cfg['dataset_mean']))
    data_loader = data.DataLoader(dataset, 32,
                                  num_workers=4,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, 0, args.cuda)
    odm_criterion = RefineMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, 0.01, args.cuda)# 0.01 -> 0.99 negative confidence threshold

    # different from normal ssd, where the PriorBox is stored inside SSD object
    priorbox = PriorBox(cfg)
    priors = Variable(priorbox.forward(), volatile=True)

    fine_tuner = PrunningFineTuner_refineDet(data_loader, arm_criterion, odm_criterion, model)

    if args.train:
        fine_tuner.train(priors, epoches = 20)
        print('Saving trained model...')
        torch.save(self.model.state_dict(), 'prunes/Refinemodel_trained.pth')
        #torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune(priors)