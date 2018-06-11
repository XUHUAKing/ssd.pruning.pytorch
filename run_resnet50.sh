#r#!/bin/bash
#su root -c 'mount -o remount,size=128G /dev/shm'
# running under youtu/akuxcwchen_pytorch:3.0
# run resnet50 on coco without evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_resnet50.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/'
# run resnet50 on coco with evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_resnet50.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/' --evaluate True
# run resnet50 on voc without evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_resnet50.py
# run resnet50 on voc with evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_voccoco_resnet50.py --evaluate True
