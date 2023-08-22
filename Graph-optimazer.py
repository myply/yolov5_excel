import copy
import csv
from treelib import Tree
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
# 创建一个graph object
G  = nx.DiGraph()
def DominatorTree(G):
    #### try to remove node i
    Domminte_nodes_dic_by_each_node = {}
    for i in G.nodes:
        print(i,G.nodes[i]['data'])
    for i in G.nodes:
        if (i!='0'):
            G1 = deepcopy(G)
            G1.remove_node(i)

            # ##if do not change the order there is no need for Topological sorting??
            ####remove the node which in degreee equal 0
            indegree_dict = {j: G1.in_degree(j) for j in G1.nodes}
            dead_node_index='0'
            no_visit_list = []
            for j in indegree_dict:
                if j!='0' and indegree_dict[j]==0:
                    dead_node_index=j
                    no_visit_list.append(dead_node_index)
                    break
            while dead_node_index!='0':
                G1.remove_node(dead_node_index)
                indegree_dict = {j: G1.in_degree(j) for j in G1.nodes}
                dead_node_index = '0'
                for j in indegree_dict:
                    if j != '0' and indegree_dict[j] == 0:
                        dead_node_index = j
                        no_visit_list.append(dead_node_index)
                        break
            # print('number of left nodes is %d after remove node %s'%(G1.number_of_nodes(),i))
            # if(i=='7'):
            #     for j in G1.nodes:
            #         print(j)
            Domminte_nodes_dic_by_each_node[i]=no_visit_list

    immediate_domminte_node_dic_by_each_node = {}
    for i in range(len(G.nodes)):
        temp = 0
        for j in range(1, i):
            ####may have many domminte node?????????
            if str(i) in Domminte_nodes_dic_by_each_node[str(j)] and j > temp:
                temp = j
        immediate_domminte_node_dic_by_each_node[i] = temp
        print("domminte node of node %d is node %d" % (i, immediate_domminte_node_dic_by_each_node[i]))

    DomT = nx.DiGraph()
    for i in immediate_domminte_node_dic_by_each_node:
        if (i==0):
            DomT.add_node(str(i),data=copy.deepcopy(G.nodes[str(i)]['data']))
        else:
            DomT.add_node(str(i), data=copy.deepcopy(G.nodes[str(i)]['data']))
            DomT.add_edge(str(immediate_domminte_node_dic_by_each_node[i]), str(i))
    return DomT




modelname='875'
filename = 'excel/'+modelname+'.csv'
data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        data.append(row)  # 选择某一列加入到data数组中
for i in range(1,len(data)):
    G.add_node(str(i-1),data=data[i])
    if(i<2):
        continue
    elif(data[i][0].strip()[0:4] == 'conv' and len(data[i][0]) > 4):
        G.add_edge(str(int(data[i][0].strip()[5:-1])-2), str(i-1))
    elif (data[i][0].strip()[0:3] == 'add' and len(data[i][0]) > 4):
        G.add_edge(str(int(data[i][0].split(' ')[1][4:]) - 2), str(i - 1))
        G.add_edge(str(int(data[i][0].split(' ')[3][0:-1]) - 2), str(i - 1))
    elif (data[i][0].strip()[0:7] == 'maxpool'):
        G.add_edge(str(int(data[i][0].strip()[8:-1])-2), str(i - 1))
    elif (data[i][0].strip()[0:3] == 'cat'):
        if(data[i][0].strip().find('-') != -1):
            G.add_edge(str(int(data[i][0].strip().split('-')[0][4:]) - 2), str(i - 1))
            G.add_edge(str(int(data[i][0].strip().split('-')[0][4:]) - 2+1), str(i - 1))
            G.add_edge(str(int(data[i][0].strip().split('-')[0][4:]) - 2+2), str(i - 1))
            G.add_edge(str(int(data[i][0].strip().split('-')[0][4:]) - 2+3), str(i - 1))
        else:
            G.add_edge(str(int(data[i][0].strip().split(' ')[0][4:]) - 2), str(i - 1))
            G.add_edge(str(int(data[i][0].strip().split(' ')[2][0:-1]) - 2), str(i - 1))
    else:
        G.add_edge(str(i-2), str(i-1))
# plt.figure(figsize=(30,30))
# # 使用对象G作为画像
# nx.draw_networkx(G)
# plt.show()

#### remove act by hand
G1=deepcopy(G)
# for i in G.nodes:
#     #### fuse conv+act
#     if G.nodes[i]['data'][0][0:4] == 'conv':
#         for j in G.neighbors(i):
#             if G.nodes[j]['data'][0]=='act':
#                 for k in G.neighbors(j):
#                     G1.add_edge(i,k)
#                 G1.remove_node(j)
#                 G1.nodes[i]['data'][13] = 1
#             #### cat have no impact on fuse
#             ####i   conv        conv
#             ####j          cat
#             ####k          act
#             ####l          node
#             elif G.nodes[j]['data'][0].strip()[0:3] == 'cat':
#                 for k in G.neighbors(j):
#                     if G.nodes[k]['data'][0] == 'act':
#                         for l in G.neighbors(k):
#                             G1.add_edge(j, l)
#                         if k in G1.nodes:
#                             G1.remove_node(k)
#                         print('rm node',k)
#                         G1.nodes[i]['data'][13] = 1
#
# index=0
# mapdict={}
# for i in G1.nodes:
#     mapdict[i]=str(index)
#     index+=1
# G1=nx.relabel_nodes(G1, mapdict)
# for i in G1.nodes:
#     ####two source
#     if G1.nodes[i]['data'][0].strip()[0:3] == 'cat' and G1.nodes[i]['data'][0].strip().find('-') != -1 :
#         for j in range(0,int(i)):
#             if i in G1.neighbors(str(j)):
#                 G1.nodes[i]['data'][14] = str(j+2)+'-'+str(j+2+3)
#                 break
#     elif G1.nodes[i]['data'][0].strip()[0:3] == 'add' or G1.nodes[i]['data'][0].strip()[0:3] == 'cat':
#         G1.nodes[i]['data'][14] = ''
#         for j in range(0, int(i)):
#             if i in G1.neighbors(str(j)):
#                 G1.nodes[i]['data'][14] += ' '+str(j+2)
#     ####singgle source
#     else:
#         for j in range(0,int(i)):
#             if i in G1.neighbors(str(j)):
#                 G1.nodes[i]['data'][14]=str(j+2)
#                 break
# #### end remove act by hand
# for i in G1.nodes:
#     print(int(i)+2,G1.nodes[i]['data'])
DomT=DominatorTree(G1)
for i in DomT.nodes:

    print(i,DomT.out_degree(i))


# index=0
# for i in G1.nodes:
#     if G1.nodes[i]['data'][0].strip()[0:4] == 'conv':
#         # G1.nodes[i]['data'][13]=1
#         G1.nodes[i]['data'].append('weight'+str(index))
#         index+=1
#
# newdata=[[' ','Ci','Co','Hi','Ho','Wi','Wo','ksize','pad','stride','M','K','N','act','src']]
# for i in G1.nodes:
#     index=G1.nodes[i]['data'][0].strip().find('(')
#     if(index!=-1):
#         G1.nodes[i]['data'][0]=G1.nodes[i]['data'][0].strip()[0:index]
#     newdata.append(G1.nodes[i]['data'])
# with open('excel/'+modelname+'_fuseact.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ')
#     for i in range(len(newdata)):
#         spamwriter.writerow(newdata[i])
