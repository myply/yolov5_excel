import copy
import csv
from treelib import Tree
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
modelname='875'
filename = 'excel/'+modelname+'.csv'
data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        data.append(row)  # 选择某一列加入到data数组中
newdata=[]
newdata.append(data[0])
for i in range(1,len(data)):
    newdata.append(data[i])
    print('1')
    print(newdata[i])
    newdata[i][3]=str(int(float(newdata[i][3])*736/960))
    newdata[i][4] = str(int(float(newdata[i][4]) * 736 / 960))
    newdata[i][5]=str(int(float(newdata[i][5])*1312/1280))
    newdata[i][6] = str(int(float(newdata[i][6]) * 1312 / 1280))
    print(newdata[i])
with open('excel/'+modelname+'_736__1312.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ')
    for i in range(len(newdata)):
        spamwriter.writerow(newdata[i])