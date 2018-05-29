#####DONE
from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

# choosing scales and aspect_ratios for default boxes here
# create default boxes
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):#cfg is the configuration of coco/voc
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):# use k feature map for prediction
            # per feature map location
            for i, j in product(range(f), repeat=2):#nested loop with (A, A) where A in range(f)
                f_k = self.image_size / self.steps[k] #size of k-th square feature map
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                # when k = 1, s_k = s_min
                s_k = self.min_sizes[k]/self.image_size # this is s_min i.e. min scale in fact
                mean += [cx, cy, s_k, s_k]# because aspect_ratio is 1, so width = height

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # when k = m, s_k = s_max
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size)) # this is s_max i.e. max scale in fact
                mean += [cx, cy, s_k_prime, s_k_prime]# because aspect_ratio is 1, so width = height

                # rest of aspect ratios other than 1 and m, which means 2 ~ m-1
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        # 6 default boxes (there are 6 [] in mean)
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
