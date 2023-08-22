import torch
import numpy as np
import torch.nn as nn
from models.experimental import attempt_load
from utils.torch_utils import ModelEMA, select_device
blocksize=[]
with open('blocksize0.1.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        linelist=line.replace('\n','').strip().split(' ')
        if linelist[0]=='block_size':
            print(linelist)
            blocksize.append(int(linelist[3]))
# print(blocksize)
print(len(blocksize))
device = select_device('')
model = attempt_load('yolov5x.pt', map_location=device)
prune_param, total_param = 0, 0
count=0
for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        count
    if isinstance(v, nn.BatchNorm2d):
        count
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        total_param += v.weight.numel()
        prune_param+=v.weight.numel()/blocksize[count]
        count+=1
print(total_param)
print(prune_param)
print(total_param/prune_param)
