#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import math

def distance(data):
    """
    计算样本之间的距离
    :param data: 样本
    :return: dis(mat)样本之间的距离
    """
    m, n = np.shape(data)
    dis = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(i, m):
            #计算i和j之间的欧式距离
            tmp = 0
            for k in range(n):
                tmp += (data[i, k] - data[j, k]) * (data[i, k] - data[j, k])
            dis[i, j] = np.sqrt(tmp)
            dis[j, i] = dis[i, j]
    return dis

def find_eps(distance_D, eps):
    """
    找到距离的《=esp的索引
    :param distance_D: 样本i与其他样本直接按的距离
    :param eps: 半径的大小
    :return: ind与样本i之间的距离《=eps的样本索引
    """
    ind = []
    n = np.shape(distance_D)[1]
    for j in range(n):
        if distance_D[0, j] <= eps:
            ind.append(j)
    return ind

def dbscan(data, eps, MinPts):
    """
    DBSCAN算法
    :param data:需要聚类的数据集
    :param eps: 半径
    :param MinPts: 半径内最少的数据点
    :return:
            types:每个样本类型，核心点，边界点，噪音点
            sub_class：每个样本所属的类别
    """
    m = np.shape(data)[0]
    # 在types中，1为核心点，0为边界点，-1为噪音点
    types = np.mat(np.zeros((1, m)))
    sub_class = np.mat(np.zeros((1, m)))
    # 用于判断该点是否处理过，0表示未处理过
    dealt = np.mat(np.zeros((m, 1)))
    # 计算每个数据点之间的距离
    dis = distance(data)
    # 用于标记类别
    number = 1

    # 对每一个点进行处理
    for i in range(m):
        # 找到未处理的点
        if dealt[i, 0] == 0:
            # 找到第i个点到其他所有点的距离
            D = dis[i,]
            # 找到半径eps内的所有点
            ind = find_eps(D, eps)
            # 区分点的类型
            # 边界点
            if len(ind) > 1 and len(ind) < MinPts + 1:
                types[0, i] = 0
                sub_class[0, i] = 0
            # 噪音点
            if len(ind) == 1:
                types[0, i] = -1
                sub_class[0, i] = -1
                dealt[i, 0] = 1
            # 核心点
            if len(ind) >= MinPts + 1:
                types[0, i] = 1
                for x in ind:
                    sub_class[0, x] = number
                # 判断核心点是否密度可达
                while len(ind) > 0:
                    dealt[ind[0], 0] = 1
                    D = dis[ind[0],]
                    tmp = ind[0]
                    del ind[0]
                    ind_1 = find_eps(D, eps)

                    if len(ind_1) > 1:  # 处理非噪音点
                        for x1 in ind_1:
                            sub_class[0, x1] = number
                        if len(ind_1) >= MinPts + 1:
                            types[0, tmp] = 1
                        else:
                            types[0, tmp] = 0

                        for j in range(len(ind_1)):
                            if dealt[ind_1[j], 0] == 0:
                                dealt[ind_1[j], 0] = 1
                                ind.append(ind_1[j])
                                sub_class[0, ind_1[j]] = number
                number += 1

    # 最后处理所有未分类的点为噪音点
    ind_2 = ((sub_class == 0).nonzero())[1]
    for x in ind_2:
        sub_class[0, x] = -1
        types[0, x] = -1

    return types, sub_class

def epsilon(data, MinPts):
    '''计算最佳半径
    input:  data(mat):训练数据
            MinPts(int):半径内的数据点的个数
    output: eps(float):半径
    '''
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps

def loadDataSet(filename):
    """
    函数说明：从文件中下载数据，并将分离除连续型变量和标签变量
    :parameter:
            data - Iris数据集
            attributes - 鸢尾花的属性
            type - 鸢尾花的类别
            sl-花萼长度 , sw-花萼宽度, pl-花瓣长度, pw-花瓣宽度
    :return:
    """
    iris_data = pd.read_csv(filename)   #打开文件
    iris_data = pd.DataFrame(data=np.array(iris_data), columns=['sl', 'sw', 'pl', 'pw', 'type'], index=range(149))   #给数据集添加列名，方便后面的操作
    attributes = iris_data[['sl', 'sw', 'pl', 'pw']]   #分离出花的属性
    iris_data['type'] = iris_data['type'].apply(lambda x: x.split('-')[1])  # 最后类别一列，感觉前面的'Iris-'有点多余即把class这一列的数据按'-'进行切分取切分后的第二个数据
    labels = iris_data['type']     #分理出花的类别
    attriLabels = []      #建立一个标签列表
    for label in labels:        #为了更方便操作，将三中不同的类型分别设为1，2，3
        if label == 'setosa':    #如果类别为setosa的话，设为1
            attriLabels.append(1)
        elif label == 'versicolor':  #如果是versicolor的时候设为2
            attriLabels.append(2)
        elif label == 'virginica':  #如果是virginica的时候设为3
            attriLabels.append(3)
    return attributes, attriLabels

import time

if __name__ == '__main__':
    time1 = time.clock()
    attributes, attriLabels = loadDataSet('Data/iris.data')
    data = np.mat(attributes)
    eps = epsilon(data, 3)
    types, sub_class = dbscan(data, eps, 3)
    time2 = time.clock()
    print(time2-time1)
    # print(sub_class)
    m = len (attributes)
    right = 0
    sub_class  = sub_class.tolist()[0]
    for lens in range(m):
       if int(sub_class[lens]) == attriLabels[lens]:
             right += 1
    a = 100*(m-right)/m
    print("错误率:",a, "%")