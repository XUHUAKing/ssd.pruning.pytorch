#!/bin/bash
su root -c 'mount -o remount,size=128G /dev/shm'

python3 train_weishi.py --dataset WEISHI --jpg_xml_path '/cephfs/person/darnellzhou/ssd.pytorch/data/train_58_0522.txt' --label_name_path '/cephfs/person/darnellzhou/ssd.pytorch/data/label58.txt' --num_workers 1   2>&1 | tee ./log_weishi.txt
