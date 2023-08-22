import pandas as pd
import numpy as np
import torch
import csv
from models.experimental import attempt_load
from utils.torch_utils import ModelEMA, select_device
model_name="yolov5s.pt"
bn=False
image_h=224
image_w=352

model=torch.load(model_name,map_location=torch.device('cpu'))['model']
device=select_device()
# model=attempt_load(model_name, map_location=device)

model.float().fuse()
# for name, param in model.named_parameters():
#     print(name)
#   0     1    2    3    4    5    6       7      8      9    10   11  12   13
headers=[' ','Ci','Co','Hi','Ho','Wi','Wo','ksize','pad','stride','M','K','N','act','src','count']
# N = (W − F + 2P )/S+1
count=2  ##easy to read in csv
model_list=[]
# for i,m in enumerate(model.model):
#     print(type(m).__name__)
for i,m in enumerate(model.model):
    if(type(m).__name__=='Conv'):
        small_model_list=[]
        ##conv
        temp=[]
        temp.append('conv')
        temp.append(m.conv.in_channels)
        temp.append(m.conv.out_channels)
        if i==0:
            temp.append(image_h)
        else:
            temp.append(model_list[i-1][-1][4])
        temp.append(int((temp[-1]-m.conv.kernel_size[0]+2*m.conv.padding[0])/m.conv.stride[0])+1)
        if i==0:
            temp.append(image_w)
        else:
            temp.append(model_list[i-1][-1][6])
        temp.append(int((temp[-1] - m.conv.kernel_size[1] + 2 * m.conv.padding[1]) / m.conv.stride[1]) + 1)
        temp.append(m.conv.kernel_size[0])
        temp.append(m.conv.padding[0])
        temp.append(m.conv.stride[0])
        temp.append(m.conv.out_channels)  ##M
        temp.append(m.conv.in_channels*m.conv.kernel_size[0]*m.conv.kernel_size[1])  ##K
        temp.append(temp[4]*temp[6])     ##N
        temp.append(0)
        temp.append(str(count-1))  ##src
        temp.append(count)
        small_model_list.append(temp)
        count += 1
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(['act',small_model_list[-1][2],small_model_list[-1][2],small_model_list[-1][4],small_model_list[-1][4],small_model_list[-1][6],small_model_list[-1][6],
                                 -1,-1,-1,-1,-1,-1,0,str(count-1),count])
        model_list.append(small_model_list)
        count += 1
        # print(model_list)
    elif (type(m).__name__ == 'Focus'):
        small_model_list = []
        #slice
        small_model_list.append(
            ['slice', 3, 12, image_h, image_h/2,
             image_w, image_w/2, -1, -1, -1, -1, -1, -1,0,str(count-1), count])
        count+=1
        ##conv
        temp = []
        temp.append('conv')
        temp.append(m.conv.conv.in_channels)
        temp.append(m.conv.conv.out_channels)
        temp.append(small_model_list[-1][4])
        temp.append(int((temp[-1] - m.conv.conv.kernel_size[0] + 2 * m.conv.conv.padding[0]) / m.conv.conv.stride[0]) + 1)
        temp.append(small_model_list[-1][6])
        temp.append(int((temp[-1] - m.conv.conv.kernel_size[1] + 2 * m.conv.conv.padding[1]) / m.conv.conv.stride[1]) + 1)
        temp.append(m.conv.conv.kernel_size[0])
        temp.append(m.conv.conv.padding[0])
        temp.append(m.conv.conv.stride[0])
        temp.append(m.conv.conv.out_channels)  ##M
        temp.append(m.conv.conv.in_channels * m.conv.conv.kernel_size[0] * m.conv.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        model_list.append(small_model_list)
        count += 1
    elif(type(m).__name__=='C3'):
        small_model_list = []
        shortcut = m.m[0].add
        ##cv1
        ##conv
        temp = []
        temp.append('conv')
        temp.append(m.cv1.conv.in_channels)
        temp.append(m.cv1.conv.out_channels)
        temp.append(model_list[i - 1][-1][4])
        temp.append(int((temp[-1] - m.cv1.conv.kernel_size[0] + 2 * m.cv1.conv.padding[0]) / m.cv1.conv.stride[0]) + 1)
        temp.append(model_list[i - 1][-1][6])
        temp.append(int((temp[-1] - m.cv1.conv.kernel_size[1] + 2 * m.cv1.conv.padding[1]) / m.cv1.conv.stride[1]) + 1)
        temp.append(m.cv1.conv.kernel_size[0])
        temp.append(m.cv1.conv.padding[0])
        temp.append(m.cv1.conv.stride[0])
        temp.append(m.cv1.conv.out_channels)  ##M
        temp.append(m.cv1.conv.in_channels * m.cv1.conv.kernel_size[0] * m.cv1.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1
        ## m
        for j in range(len(m.m)):
            ##m.cv1 use cv1 result
            temp = []
            temp.append('conv')
            temp.append(m.m[j].cv1.conv.in_channels)
            temp.append(m.m[j].cv1.conv.out_channels)
            #use cv1 result
            temp.append(small_model_list[-1][4])
            temp.append(
                int((temp[-1] - m.m[j].cv1.conv.kernel_size[0] + 2 * m.m[j].cv1.conv.padding[0]) / m.m[j].cv1.conv.stride[0]) + 1)
            temp.append(small_model_list[-1][6])
            temp.append(
                int((temp[-1] - m.m[j].cv1.conv.kernel_size[1] + 2 * m.m[j].cv1.conv.padding[1]) / m.m[j].cv1.conv.stride[1]) + 1)
            temp.append(m.m[j].cv1.conv.kernel_size[0])
            temp.append(m.m[j].cv1.conv.padding[0])
            temp.append(m.m[j].cv1.conv.stride[0])
            temp.append(m.m[j].cv1.conv.out_channels)  ##M
            temp.append(m.m[j].cv1.conv.in_channels * m.m[j].cv1.conv.kernel_size[0] * m.m[j].cv1.conv.kernel_size[1])  ##K
            temp.append(temp[4] * temp[6])  ##N
            temp.append(0)
            temp.append(count - 1)
            temp.append(count)
            count += 1
            small_model_list.append(temp)
            ## bn
            if bn:
                small_model_list.append(
                    ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                     small_model_list[-1][4],
                     small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,count])
                count += 1
            ##act
            small_model_list.append(
                ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                 small_model_list[-1][4],
                 small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
            count += 1

            ##m.cv2 use cv1 result
            temp = []
            temp.append('conv')
            temp.append(m.m[j].cv2.conv.in_channels)
            temp.append(m.m[j].cv2.conv.out_channels)
            # use m.cv1 result
            temp.append(small_model_list[-1][4])
            temp.append(
                int((temp[-1] - m.m[j].cv2.conv.kernel_size[0] + 2 * m.m[j].cv2.conv.padding[0]) / m.m[j].cv2.conv.stride[0]) + 1)
            temp.append(small_model_list[-1][6])
            temp.append(
                int((temp[-1] - m.m[j].cv2.conv.kernel_size[1] + 2 * m.m[j].cv2.conv.padding[1]) / m.m[j].cv2.conv.stride[1]) + 1)
            temp.append(m.m[j].cv2.conv.kernel_size[0])
            temp.append(m.m[j].cv2.conv.padding[0])
            temp.append(m.m[j].cv2.conv.stride[0])
            temp.append(m.m[j].cv2.conv.out_channels)  ##M
            temp.append(m.m[j].cv2.conv.in_channels * m.m[j].cv2.conv.kernel_size[0] * m.m[j].cv2.conv.kernel_size[1])  ##K
            temp.append(temp[4] * temp[6])  ##N
            temp.append(0)
            temp.append(count - 1)
            temp.append(count)
            count += 1
            small_model_list.append(temp)
            ## bn
            if bn:
                small_model_list.append(
                    ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                     small_model_list[-1][4],
                     small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,count])
                count += 1
            ##act
            small_model_list.append(
                ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                 small_model_list[-1][4],
                 small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0,str(count-1), count])
            count += 1

            if shortcut:
                if bn:
                    small_model_list.append(['add('+str(small_model_list[-7][-1])+' '+str(small_model_list[-1][-1])+')',small_model_list[-1][1],small_model_list[-1][2],small_model_list[-1][3],small_model_list[-1][4],small_model_list[-1][5],small_model_list[-1][6],
                                             -1, -1, -1, -1, -1,-1, 0,str(small_model_list[-7][-1])+' '+str(small_model_list[-1][-1]),count])
                else:
                    small_model_list.append(
                        ['add(' + str(small_model_list[-5][-1]) + ' ' + str(small_model_list[-1][-1]) + ')',
                         small_model_list[-1][1], small_model_list[-1][2], small_model_list[-1][3],
                         small_model_list[-1][4], small_model_list[-1][5], small_model_list[-1][6], -1, -1, -1, -1, -1,
                         -1,0,str(small_model_list[-5][-1]) + ' ' + str(small_model_list[-1][-1]), count])
                count+=1
        ##cv2 conv
        temp = []
        temp.append('conv(' + str(model_list[i - 1][-1][-1]) + ')')
        temp.append(m.cv2.conv.in_channels)
        temp.append(m.cv2.conv.out_channels)
        temp.append(model_list[i - 1][-1][4])
        temp.append(int((temp[-1] - m.cv2.conv.kernel_size[0] + 2 * m.cv2.conv.padding[0]) / m.cv2.conv.stride[0]) + 1)
        temp.append(model_list[i - 1][-1][6])
        temp.append(int((temp[-1] - m.cv2.conv.kernel_size[1] + 2 * m.cv2.conv.padding[1]) / m.cv2.conv.stride[1]) + 1)
        temp.append(m.cv2.conv.kernel_size[0])
        temp.append(m.cv2.conv.padding[0])
        temp.append(m.cv2.conv.stride[0])
        temp.append(m.cv2.conv.out_channels)  ##M
        temp.append(m.cv2.conv.in_channels * m.cv2.conv.kernel_size[0] * m.cv2.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(str(model_list[i - 1][-1][-1]))
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
             small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1

        ##cat
        if bn:
            small_model_list.append(
                ['cat(' + str(small_model_list[-4][-1]) + ' ' + str(small_model_list[-1][-1]) + ')',
                 small_model_list[-1][2], small_model_list[-1][2] + small_model_list[-3][2], small_model_list[-1][4],
                 small_model_list[-1][4],
                 small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])

        else:
            small_model_list.append(
            ['cat('+str(small_model_list[-3][-1])+' '+str(small_model_list[-1][-1])+')', small_model_list[-1][2], small_model_list[-1][2]+small_model_list[-3][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,str(small_model_list[-3][-1])+' '+str(small_model_list[-1][-1]),count])
        count += 1

        ##cv3 conv
        temp = []
        temp.append('conv')
        temp.append(m.cv3.conv.in_channels)
        temp.append(m.cv3.conv.out_channels)
        temp.append(small_model_list[-1][4])

        temp.append(int((temp[-1] - m.cv3.conv.kernel_size[0] + 2 * m.cv3.conv.padding[0]) / m.cv3.conv.stride[0]) + 1)
        temp.append(small_model_list[-1][6])
        temp.append(int((temp[-1] - m.cv3.conv.kernel_size[1] + 2 * m.cv3.conv.padding[1]) / m.cv3.conv.stride[1]) + 1)
        temp.append(m.cv3.conv.kernel_size[0])
        temp.append(m.cv3.conv.padding[0])
        temp.append(m.cv3.conv.stride[0])
        temp.append(m.cv3.conv.out_channels)  ##M
        temp.append(m.cv3.conv.in_channels * m.cv3.conv.kernel_size[0] * m.cv3.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
             small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1

        model_list.append(small_model_list)
        # break
    elif (type(m).__name__ == 'BottleneckCSP'):
        small_model_list = []
        shortcut = m.m[0].add
        ##cv1 Conv
        temp = []
        temp.append('conv')
        temp.append(m.cv1.conv.in_channels)
        temp.append(m.cv1.conv.out_channels)
        temp.append(model_list[i - 1][-1][4])
        temp.append(int((temp[-1] - m.cv1.conv.kernel_size[0] + 2 * m.cv1.conv.padding[0]) / m.cv1.conv.stride[0]) + 1)
        temp.append(model_list[i - 1][-1][6])
        temp.append(int((temp[-1] - m.cv1.conv.kernel_size[1] + 2 * m.cv1.conv.padding[1]) / m.cv1.conv.stride[1]) + 1)
        temp.append(m.cv1.conv.kernel_size[0])
        temp.append(m.cv1.conv.padding[0])
        temp.append(m.cv1.conv.stride[0])
        temp.append(m.cv1.conv.out_channels)  ##M
        temp.append(m.cv1.conv.in_channels * m.cv1.conv.kernel_size[0] * m.cv1.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1

        ## m
        for j in range(len(m.m)):
            ##m.cv1 use cv1 result
            temp = []
            temp.append('conv')
            temp.append(m.m[j].cv1.conv.in_channels)
            temp.append(m.m[j].cv1.conv.out_channels)
            # use cv1 result
            temp.append(small_model_list[-1][4])
            temp.append(
                int((temp[-1] - m.m[j].cv1.conv.kernel_size[0] + 2 * m.m[j].cv1.conv.padding[0]) / m.m[j].cv1.conv.stride[0]) + 1)
            temp.append(small_model_list[-1][6])
            temp.append(
                int((temp[-1] - m.m[j].cv1.conv.kernel_size[1] + 2 * m.m[j].cv1.conv.padding[1]) / m.m[j].cv1.conv.stride[1]) + 1)
            temp.append(m.m[j].cv1.conv.kernel_size[0])
            temp.append(m.m[j].cv1.conv.padding[0])
            temp.append(m.m[j].cv1.conv.stride[0])
            temp.append(m.m[j].cv1.conv.out_channels)  ##M
            temp.append(m.m[j].cv1.conv.in_channels * m.m[j].cv1.conv.kernel_size[0] * m.m[j].cv1.conv.kernel_size[1])  ##K
            temp.append(temp[4] * temp[6])  ##N
            temp.append(0)
            temp.append(count - 1)
            temp.append(count)
            count += 1
            small_model_list.append(temp)
            ## bn
            if bn:
                small_model_list.append(
                    ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                     small_model_list[-1][4],
                     small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,count])
                count += 1
            ##act
            small_model_list.append(
                ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                 small_model_list[-1][4],
                 small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
            count += 1

            ##m.cv2 use cv1 result
            temp = []
            temp.append('conv')
            temp.append(m.m[j].cv2.conv.in_channels)
            temp.append(m.m[j].cv2.conv.out_channels)
            # use m.cv1 result
            temp.append(small_model_list[-1][4])
            temp.append(
                int((temp[-1] - m.m[j].cv2.conv.kernel_size[0] + 2 * m.m[j].cv2.conv.padding[0]) / m.m[j].cv2.conv.stride[0]) + 1)
            temp.append(small_model_list[-1][6])
            temp.append(
                int((temp[-1] - m.m[j].cv2.conv.kernel_size[1] + 2 * m.m[j].cv2.conv.padding[1]) / m.m[j].cv2.conv.stride[1]) + 1)
            temp.append(m.m[j].cv2.conv.kernel_size[0])
            temp.append(m.m[j].cv2.conv.padding[0])
            temp.append(m.m[j].cv2.conv.stride[0])
            temp.append(m.m[j].cv2.conv.out_channels)  ##M
            temp.append(m.m[j].cv2.conv.in_channels * m.m[j].cv2.conv.kernel_size[0] * m.m[j].cv2.conv.kernel_size[1])  ##K
            temp.append(temp[4] * temp[6])  ##N
            temp.append(0)
            temp.append(count - 1)
            temp.append(count)
            count += 1
            small_model_list.append(temp)
            ## bn
            if bn:
                small_model_list.append(
                    ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                     small_model_list[-1][4],
                     small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,count])
                count += 1
            ##act
            small_model_list.append(
                ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4],
                 small_model_list[-1][4],
                 small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0,str(count-1),count])
            count += 1

            if shortcut:
                if bn:
                    small_model_list.append(['add('+str(small_model_list[-7][-1])+' '+str(small_model_list[-1][-1])+')',small_model_list[-1][1],small_model_list[-1][2],small_model_list[-1][3],small_model_list[-1][4],small_model_list[-1][5],small_model_list[-1][6],
                                             -1, -1, -1, -1, -1,-1, str(small_model_list[-7][-1])+' '+str(small_model_list[-1][-1]),count])
                else:
                    small_model_list.append(
                        ['add(' + str(small_model_list[-5][-1]) + ' ' + str(small_model_list[-1][-1]) + ')',
                         small_model_list[-1][1], small_model_list[-1][2], small_model_list[-1][3],
                         small_model_list[-1][4], small_model_list[-1][5], small_model_list[-1][6], -1, -1, -1, -1, -1,
                         -1,0,str(small_model_list[-5][-1]) + ' ' + str(small_model_list[-1][-1]),count])
                count+=1

        ##cv3 conv
        temp = []
        temp.append('conv')
        temp.append(m.cv3.in_channels)
        temp.append(m.cv3.out_channels)
        temp.append(small_model_list[-1][4])
        temp.append(int((temp[-1] - m.cv3.kernel_size[0] + 2 * m.cv3.padding[0]) / m.cv3.stride[0]) + 1)
        temp.append(small_model_list[-1][6])
        temp.append(int((temp[-1] - m.cv3.kernel_size[1] + 2 * m.cv3.padding[1]) / m.cv3.stride[1]) + 1)
        temp.append(m.cv3.kernel_size[0])
        temp.append(m.cv3.padding[0])
        temp.append(m.cv3.stride[0])
        temp.append(m.cv3.out_channels)  ##M
        temp.append(m.cv3.in_channels * m.cv3.kernel_size[0] * m.cv3.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)

        ##cv2 conv
        temp = []
        temp.append('conv('+str(model_list[i - 1][-1][-1])+')')
        temp.append(m.cv2.in_channels)
        temp.append(m.cv2.out_channels)
        temp.append(model_list[i - 1][-1][4])
        temp.append(int((temp[-1] - m.cv2.kernel_size[0] + 2 * m.cv2.padding[0]) / m.cv2.stride[0]) + 1)
        temp.append(model_list[i - 1][-1][6])
        temp.append(int((temp[-1] - m.cv2.kernel_size[1] + 2 * m.cv2.padding[1]) / m.cv2.stride[1]) + 1)
        temp.append(m.cv2.kernel_size[0])
        temp.append(m.cv2.padding[0])
        temp.append(m.cv2.stride[0])
        temp.append(m.cv2.out_channels)  ##M
        temp.append(m.cv2.in_channels * m.cv2.kernel_size[0] * m.cv2.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(str(model_list[i - 1][-1][-1]))
        temp.append(count)
        count += 1
        small_model_list.append(temp)


        ##cat

        small_model_list.append(
            ['cat('+str(small_model_list[-2][-1])+' '+str(small_model_list[-1][-1])+')', small_model_list[-1][2], small_model_list[-1][2]+small_model_list[-2][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(small_model_list[-2][-1])+' '+str(small_model_list[-1][-1]),count])
        count += 1
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1

        ##cv4 conv
        temp = []
        temp.append('conv')
        temp.append(m.cv4.conv.in_channels)
        temp.append(m.cv4.conv.out_channels)
        temp.append(small_model_list[-1][4])
        temp.append(int((temp[-1] - m.cv4.conv.kernel_size[0] + 2 * m.cv4.conv.padding[0]) / m.cv4.conv.stride[0]) + 1)
        temp.append(small_model_list[-1][6])
        temp.append(int((temp[-1] - m.cv4.conv.kernel_size[1] + 2 * m.cv4.conv.padding[1]) / m.cv4.conv.stride[1]) + 1)
        temp.append(m.cv4.conv.kernel_size[0])
        temp.append(m.cv4.conv.padding[0])
        temp.append(m.cv4.conv.stride[0])
        temp.append(m.cv4.conv.out_channels)  ##M
        temp.append(m.cv4.conv.in_channels * m.cv4.conv.kernel_size[0] * m.cv4.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1
        model_list.append(small_model_list)
    elif (type(m).__name__ == 'SPPF' or type(m).__name__ == 'SPP'):
        # break
        ##SPPF cv1
        small_model_list=[]
        temp = []
        temp.append('conv')
        temp.append(m.cv1.conv.in_channels)
        temp.append(m.cv1.conv.out_channels)
        temp.append(model_list[i - 1][-1][4])
        temp.append(int((temp[-1] - m.cv1.conv.kernel_size[0] + 2 * m.cv1.conv.padding[0]) / m.cv1.conv.stride[0]) + 1)
        temp.append(model_list[i - 1][-1][6])
        temp.append(int((temp[-1] - m.cv1.conv.kernel_size[1] + 2 * m.cv1.conv.padding[1]) / m.cv1.conv.stride[1]) + 1)
        temp.append(m.cv1.conv.kernel_size[0])
        temp.append(m.cv1.conv.padding[0])
        temp.append(m.cv1.conv.stride[0])
        temp.append(m.cv1.conv.out_channels)  ##M
        temp.append(m.cv1.conv.in_channels * m.cv1.conv.kernel_size[0] * m.cv1.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(count-1),count])
        count += 1
        ##SPPF maxPOOL
        for j in range(3):
            if(type(m).__name__ == 'SPPF'):
                small_model_list.append(
            ['maxpool('+str(small_model_list[-1][-1])+')', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], m.m.kernel_size,  m.m.padding,  m.m.stride, -1, -1, -1, 0,str(small_model_list[-1][-1]),count])
            else:
                small_model_list.append(
                    ['maxpool(' + str(small_model_list[-j-1][-1]) + ')', small_model_list[-1][2], small_model_list[-1][2],
                     small_model_list[-1][4], small_model_list[-1][4],
                     small_model_list[-1][6], small_model_list[-1][6], m.m[j].kernel_size, m.m[j].padding, m.m[j].stride, -1, -1,
                     -1,0, str(small_model_list[-j-1][-1]),count])
            count += 1
        ##cat

        small_model_list.append(
            ['cat('+str(small_model_list[-4][-1])+'-'+str(small_model_list[-1][-1])+')', small_model_list[-1][2], small_model_list[-1][2]+small_model_list[-2][2]+small_model_list[-3][2]+small_model_list[-4][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, str(small_model_list[-4][-1])+'-'+str(small_model_list[-1][-1]),count])
        count += 1
        ##SPPF cv2
        temp = []
        temp.append('conv')
        temp.append(m.cv2.conv.in_channels)
        temp.append(m.cv2.conv.out_channels)
        temp.append(small_model_list[-1][4])
        temp.append(int((temp[-1] - m.cv2.conv.kernel_size[0] + 2 * m.cv2.conv.padding[0]) / m.cv2.conv.stride[0]) + 1)
        temp.append(small_model_list[-1][6])
        temp.append(int((temp[-1] - m.cv2.conv.kernel_size[1] + 2 * m.cv2.conv.padding[1]) / m.cv2.conv.stride[1]) + 1)
        temp.append(m.cv2.conv.kernel_size[0])
        temp.append(m.cv2.conv.padding[0])
        temp.append(m.cv2.conv.stride[0])
        temp.append(m.cv2.conv.out_channels)  ##M
        temp.append(m.cv2.conv.in_channels * m.cv2.conv.kernel_size[0] * m.cv2.conv.kernel_size[1])  ##K
        temp.append(temp[4] * temp[6])  ##N
        temp.append(0)
        temp.append(count - 1)
        temp.append(count)
        count += 1
        small_model_list.append(temp)
        ## bn
        if bn:
            small_model_list.append(
            ['bn', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1,0, count])
            count += 1
        ##act
        small_model_list.append(
            ['act', small_model_list[-1][2], small_model_list[-1][2], small_model_list[-1][4], small_model_list[-1][4],
             small_model_list[-1][6], small_model_list[-1][6], -1, -1, -1, -1, -1, -1, 0,str(count-1),count])
        count += 1
        model_list.append(small_model_list)
    elif (type(m).__name__ == 'Upsample'):
        ##act
        small_model_list=[]
        print('upsampling',m.mode)
        small_model_list.append(
            ['upsampling', model_list[-1][-1][2], model_list[-1][-1][2], model_list[-1][-1][4], model_list[-1][-1][4]*m.scale_factor,
             model_list[-1][-1][6], model_list[-1][-1][6]*m.scale_factor, -1, -1, -1, -1, -1, -1,0,str(count-1), count])
        count += 1
        model_list.append(small_model_list)
    elif (type(m).__name__ == 'Concat'):
        small_model_list = []
        small_model_list.append(
            ['cat('+str(model_list[-1][-1][-1])+' '+str(model_list[m.f[1]][-1][-1])+')', model_list[-1][-1][2], model_list[-1][-1][2]*2, model_list[-1][-1][4],
             model_list[-1][-1][4],
             model_list[-1][-1][6], model_list[-1][-1][6], -1, -1, -1, -1, -1, -1, 0,str(model_list[-1][-1][-1])+' '+str(model_list[m.f[1]][-1][-1]),count])
        count += 1
        model_list.append(small_model_list)
    elif (type(m).__name__ == 'Detect'):
        small_model_list = []
        for j in range(len(m.m)):
            temp = []
            temp.append('conv('+str(model_list[m.f[j]][-1][-1])+')')
            temp.append(m.m[j].in_channels)
            temp.append(m.m[j].out_channels)
            temp.append(model_list[m.f[j]][-1][4])
            temp.append(
                int((temp[-1] - m.m[j].kernel_size[0] + 2 *m.m[j].padding[0]) / m.m[j].stride[0]) + 1)
            temp.append(model_list[m.f[j]][-1][6])
            temp.append(
                int((temp[-1] - m.m[j].kernel_size[1] + 2 * m.m[j].padding[1]) / m.m[j].stride[1]) + 1)
            temp.append(m.m[j].kernel_size[0])
            temp.append(m.m[j].padding[0])
            temp.append(m.m[j].stride[0])
            temp.append(m.m[j].out_channels)  ##M
            temp.append(m.m[j].in_channels * m.m[j].kernel_size[0] * m.m[j].kernel_size[1])  ##K
            temp.append(temp[4] * temp[6])  ##N
            temp.append(0)
            temp.append(str(model_list[m.f[j]][-1][-1]))
            temp.append(count)
            count += 1
            small_model_list.append(temp)
        model_list.append(small_model_list)
    else:
        print('unknow module:'+type(m).__name__)
print(count)
# for i in small_model_list:
#     print(i)
# print(small_model_list[-4])
# print(small_model_list[-3])
# print(small_model_list[-2])
# print(small_model_list[-1])

if bn:
    with open('excel_bn/' + model_name[0:-3] + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ')
        spamwriter.writerow(headers[0:-1])
        for i in range(len(model_list)):
            for j in model_list[i]:
                spamwriter.writerow(j[0:-1])
else:
    with open('excel/' + model_name[0:-3] + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ')
        spamwriter.writerow(headers[0:-1])
        for i in range(len(model_list)):
            for j in model_list[i]:
                spamwriter.writerow(j[0:-1])



# for i in range(len(model_list)):
#     for j in model_list[i]:
#         print(j)