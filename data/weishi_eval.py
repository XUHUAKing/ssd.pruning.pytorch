# --------------------------------
# eval tools for WEISHI dataset or other jpg-xml path like dataset
# --------------------------------

import pickle
import xml.etree.ElementTree as ET

import numpy as np
import os

def parse_rec(filename):
    """ Parse a WEISHI xml file """
    tree = ET.parse(filename) #filename is the path
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text.lower().strip()
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def weishi_ap(rec, prec, use_07_metric=True):
    """
    ap = weishi_ap(rec, prec, [use_07_metric])
    Compute WEISHI AP given precision and recall.
    If use_07_metric is true, uses the VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1): # [0, 0.1, 0.2, 0.3 ..., 1]
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# for a specific class
def weishi_eval(detpath,
                dataset,
                classname,
                cachedir,
                ovthresh=0.5,
                use_07_metric=True):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    imagenames = dataset.ids# a list of image ids
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        # read list of images
        fin = open(dataset.image_xml_path, 'r')
        for i, line in enumerate(fin.readlines()):
            line = line.strip()
            des = line.split(' ')
            annopath = des[1]
            imagename = imagenames[i]
            recs[imagename] = parse_rec(annopath)
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    # recs stores the annots for each images
    # class_recs stores the gt for a class

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # go through every image
    for imagename in imagenames:
        # and extract those objects in this image that are under this designated class
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R]) # the object belongs to this class
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # for each image, store the bboxs for this class inside this image
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines] # [[image_id1, confidence1, xmin1, xmax1, ymin1, ymax1], [], [], []...]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)# find the vertex of intersected rectangle
                ih = np.maximum(iymax - iymin, 0.)# find the vertex of intersected rectangle
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:# ignore difficult
                    if not R['det'][jmax]: #R['det'][jmax] has NOT already been 1
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1. # false positive
            else:
                fp[d] = 1. #false positive

        # compute precision recall
        fp = np.cumsum(fp)# how many 1 in fp array
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = weishi_ap(rec, prec, use_07_metric)
    else:
        print("Exception: line == 1!")
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap
