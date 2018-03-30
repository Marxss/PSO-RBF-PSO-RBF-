# coding=utf-8

# 最大最小距离算法的Python实现
# 数据集形式data=[[],[],...,[]]
# 聚类结果形式result=[[[],[],...],[[],[],...],...]
# 其中[]为一个模式样本，[[],[],...]为一个聚类

import math
from scipy.linalg import norm
from numpy import *


def start_cluster(data, t):
    zs = [data[0]]  # 聚类中心集，选取第一个模式样本作为第一个聚类中心Z1
    # 第2步：寻找Z2,并计算阈值T
    T = step2(data, t, zs)
    # 第3,4,5步，寻找所有的聚类中心
    get_clusters(data, zs, T)
    # 按最近邻分类
    result = classify(data, zs, T)
    return result


# 分类
def classify(data, zs, T):
    result = [[] for i in range(len(zs))]
    for aData in data:
        min_distance = T
        index = 0
        for i in range(len(zs)):
            temp_distance = get_distance(aData, zs[i])
            if temp_distance < min_distance:
                min_distance = temp_distance
                index = i
        result[index].append(aData)
    return result


# 寻找所有的聚类中心
def get_clusters(data, zs, T):
    max_min_distance = 0
    index = 0
    for i in range(len(data)):
        min_distance = []
        for j in range(len(zs)):
            distance = get_distance(data[i], zs[j])
            min_distance.append(distance)
        min_dis = min(dis for dis in min_distance)
        if min_dis > max_min_distance:
            max_min_distance = min_dis
            index = i
    if max_min_distance > T:
        zs.append(data[index])
        # 迭代
        get_clusters(data, zs, T)


# 寻找Z2,并计算阈值T
def step2(data, t, zs):
    distance = 0
    index = 0
    for i in range(len(data)):
        temp_distance = get_distance(data[i], zs[0])
        if temp_distance > distance:
            distance = temp_distance
            index = i
    # 将Z2加入到聚类中心集中
    zs.append(data[index])
    # 计算阈值T
    T = t * distance
    return T


# 计算两个模式样本之间的欧式距离
def get_distance(data1, data2):
    data1=array(data1)
    data2 = array(data2)
    n=norm(data1-data2)
    return n

if __name__ == '__main__':
    data = [[0, 0], [3, 8], [1, 1], [2, 2], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]]
    t = 0.1
    result = start_cluster(data, t)
    for i in range(len(result)):
        print ("----------第" + str(i+1) + "个聚类----------")
        print (result[i])