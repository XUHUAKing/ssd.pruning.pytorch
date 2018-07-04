# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# ---------------- training dataset paths
# VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
VOC_ROOT = '/cephfs/share/data/VOCdevkit/'
#COCO_ROOT = osp.join(HOME, 'data/coco/')
COCO_ROOT = '/cephfs/share/data/coco_xy/'

# ---------------- testing dataset paths
"""FOR VOC"""
# the val dataset root
voc_val_dataset_root = '/cephfs/share/data/VOCdevkit/'
#annopath = os.path.join(voc_val_dataset_root, 'VOC2007', 'Annotations', '%s.xml')
#imgpath = os.path.join(voc_val_dataset_root, 'VOC2007', 'JPEGImages', '%s.jpg')
#imgsetpath = os.path.join(voc_val_dataset_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')

"""FOR COCO"""
# the val dataset root
coco_val_dataset_root = '/cephfs/share/data/coco_xy/'

"""FOR WEISHI"""
weishi_val_dataset_root = '/cephfs/share/data/weishi_xh'
weishi_val_imgxml_path = '/cephfs/share/data/weishi_xh/val_57_0511_new.txt'# be imported and used in train_xx.py file

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    #'lr_steps': (80000, 100000, 120000),
    #'max_iter': 120000,
    'dataset_mean':(104, 117, 123),
    'testset_mean':(104, 117, 123),
    'max_epoch': 10000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300, # size of image
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# for refineDet
voc320 = {
    'num_classes': 21,
    'dataset_mean':(104, 117, 123),
    'testset_mean':(104, 117, 123),
    'max_epoch': 10000,
    'feature_maps': [40, 20, 10, 5],
    'min_dim': 320, # size of image
    'steps': [8, 16, 32, 64],# image_size/steps[k] = the size for kth feature map
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC320',
}

coco = {
    'num_classes': 201,
    #'lr_steps': (280000, 360000, 400000),
    #'max_iter': 400000,
    'dataset_mean':(104, 117, 123),
    'testset_mean':(104, 117, 123),
    'max_epoch': 40000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300, # size of image
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

weishi = {
    'num_classes': 58,
    #'lr_steps': (280000, 360000, 400000),
    #'max_iter': 400000,
    'dataset_mean':(114, 114, 114),
    'testset_mean':(114, 114, 114),
    'max_epoch': 40000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300, # size of image
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'WEISHI',
}
