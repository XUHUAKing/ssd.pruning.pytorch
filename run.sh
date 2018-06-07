#!/bin/bash
su root -c 'mount -o remount,size=128G /dev/shm'

python3 train.py --dataset VOC --dataset_root '/cephfs/share/data/VOCdevkit/'  2>&1 | tee ./log_ssd_o.txt
