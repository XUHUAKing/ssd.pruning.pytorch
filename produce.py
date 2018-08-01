import torch
from collections import OrderedDict

state_dict = torch.load('mobilev2_withfc.pth')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
#    head = k[:7]
#    if head == 'module.':
#        name = k[7:]
#    else:
#        name = k

#    head = k[:6]
#    if head == 'model.':
#        name = 'features.' + k[6:]
#    else:
#        name = k

#    head = k[:3]
#    if head == 'fc.':
#        name = 'classifier.' + k[3:]
#    else:
#       name = k
#    new_state_dict[name] = v
    print(v)

#torch.save(new_state_dict, 'mobilev2_reducedfc.pth')
