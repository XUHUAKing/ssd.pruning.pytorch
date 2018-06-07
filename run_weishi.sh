#!/bin/bash
su root -c 'mount -o remount,size=128G /dev/shm'

#python3 train_weishi.py --dataset WEISHI --jpg_xml_path '/cephfs/share/data/train_58_0522.txt' --label_name_path '/cephfs/person/darnellzhou/ssd.pytorch/data/label58.txt' --num_workers 8   2>&1 | tee ./log_weishi.txt
python3 train_weishi.py --dataset WEISHI --jpg_xml_path '/cephfs/person/darnellzhou/ssd.pytorch/data_history/val_57_0511_new_target.txt' --label_name_path '/cephfs/person/darnellzhou/ssd.pytorch/data/label58.txt' --num_workers 8   2>&1 | tee ./log_weishi.txt
