'''
run_vgg.sh
'''
#r#!/bin/bash
#su root -c 'mount -o remount,size=128G /dev/shm'
# running under youtu/akuxcwchen_pytorch:3.0
# run vgg on coco without evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_vgg.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/'
# run vgg on coco with evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_vgg.py --dataset COCO --dataset_root '/cephfs/share/data/coco_xy/' --evaluate True
# run vgg on voc without evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_vgg.py
# run vgg on voc with evaluation
cd /cephfs/person/xuhuahuang/ssd.pytorch.tencent && python3 train_vgg.py --evaluate True

'''
run_resnet50.sh
'''
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

'''
run_weishi.sh
'''
#!/bin/bash
su root -c 'mount -o remount,size=128G /dev/shm'

#python3 train_weishi.py --dataset WEISHI --jpg_xml_path '/cephfs/share/data/train_58_0522.txt' --label_name_path '/cephfs/person/darnellzhou/ssd.pytorch/data/label58.txt' --num_workers 8   2>&1 | tee ./log_weishi.txt
python3 train_vgg.py --dataset WEISHI --jpg_xml_path '/cephfs/share/data/weishi_xh/train_58_0522.txt' --label_name_path '/cephfs/share/data/weishi_xh/label58.txt' --num_workers 8
