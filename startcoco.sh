#!/bin/bash
su root -c 'mount -o remount,size=128G /dev/shm'
cd /cephfs/person/xuhuahuang/ssd.pytorch-master && python3 train.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/'
