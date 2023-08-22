import pandas as pd
import numpy as np
import torch
import csv
from models.experimental import attempt_load
from utils.torch_utils import ModelEMA, select_device
modelname='875'
filename = 'excel/'+modelname+'.csv'
data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        data.append(row)  # 选择某一列加入到data数组中
index=0
for i in range(len(data)):
    print(i)
    if data[i][0].strip()[0:4] == 'conv':
        data[i].append("weight"+str(index))
        index+=1
with open('excel/'+modelname+'_fuseact.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ')
    for i in range(len(data)):
        spamwriter.writerow(data[i])
