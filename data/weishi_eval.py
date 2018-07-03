# --------------------------------
# eval tools for WEISHI dataset
# --------------------------------

"""FOR WEISHI"""
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
        ap = cal_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap
    
'''
def do_python_eval(dataset, output_dir='output', use_07=True):
    cachedir = os.path.join(dataset_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        #filename = get_voc_results_file_template(set_type, cls)
        filename = get_results_file_template(set_type, cls)
        if gset == 'weishi':
            rec, prec, ap = weishi_eval(
                filename, dataset, cls, cachedir,
                ovthresh=0.5, use_07_metric=use_07_metric)
'''
