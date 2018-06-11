r#!/bin/bash
su root -c 'mount -o remount,size=128G /dev/shm'
# run vgg on coco without evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_vgg.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/'
# run vgg on coco with evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_vgg.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/' --evaluate True
# run vgg on voc without evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_vgg.py
# run vgg on voc with evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_vgg.py --evaluate True
