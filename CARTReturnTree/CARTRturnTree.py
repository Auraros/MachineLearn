
# God Bless No Bugs!
#  *
#  *                1      _ooOoo_        1
#  *                 1    o8888888o     1
#  *                   1  88" . "88   1
#  *                      (| -_- |)
#  *                      O\  =  /O
#  *                   ____/`---'\____
#  *                 .'  \\|     |//  `.
#  *                /  \\|||  :  |||//  \
#  *               /  _||||| -:- |||||-  \
#  *               |   | \\\  -  /// |   |
#  *               | \_|  ''\---/''  |   |
#  *               \  .-\__  `-`  ___/-. /
#  *             ___`. .'  /--.--\  `. . __
#  *          ."" '<  `.___\_<|>_/___.'  >'"".
#  *         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#  *         \  \ `-.   \_ __\ /__ _/   .-` /  /
#  *    ======`-.____`-.___\_____/___.-`____.-'======
#  *                       `=---='
#  *    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



# -*- coding:utf-8 -*-
import numpy as np
from random import randint
import pandas as pd
import random

def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
    Website:
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 转化为float类型
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差

    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    函数说明:找到数据的最佳二元切分方式函数
    Parameters:
        dataSet - 数据集合
        leafType - 生成叶结点
        regErr - 误差估计函数
        ops - 用户定义的参数构成的元组
    Returns:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
    """
    import types
    # tolS允许的误差下降值,tolN切分的最少样本数
    tolS = ops[0];
    tolN = ops[1]
    # 如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 统计数据集合的行m和列n
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = float('inf');
    bestIndex = 0;
    bestValue = 0
    # 遍历所有特征列
    for featIndex in range(n - 1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN,则退出
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            # 计算误差估计
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue


def createTree(dataSet, labels, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树
    """
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # r如果没有特征,则返回特征值
    if feat == None: return val
    # 回归树
    retTree = {}
    n = labels[feat]
    retTree['spInd'] = n
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet, labels, leafType, errType, ops)
    retTree['right'] = createTree(rSet, labels, leafType, errType, ops)
    return retTree

def createTree1(dataSet, labels, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树
    """
    # 选择最佳切分特征和特征值
    m = len(labels)-1
    random_number = randint(0, m)
    feat = random_number
    dataSet = pd.DataFrame(dataSet)
    max_number = dataSet.loc[:,feat].max()
    min_nuber = dataSet.loc[:,feat].min()
    val = random.uniform(min_nuber, max_number)
    # feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    dataSet = np.mat(dataSet.values.tolist())
    # r如果没有特征,则返回特征值
    if feat == None:
        return val
    # 回归树
    retTree = {}
    n = labels[feat]
    retTree['spInd'] = n
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet, labels, leafType, errType, ops)
    retTree['right'] = createTree(rSet, labels, leafType, errType, ops)
    return retTree

def classify(inputTree,  testVec, labels):
    """
    获取分类的结果（算法的使用器）
    :param inputTree:决策树字典
    :param testVec: 测试向量
    :return:
    """
    #获取根节点的名称，并且把根节点变成列表
    firstSide = list(inputTree.keys())
    #得到变量的索引和值
    spInd_name = firstSide[0]
    spVal_name = firstSide[1]
    spName = inputTree[spInd_name]
    spVal = inputTree[spVal_name]
    spInd = labels.index(spName)
    if float(testVec[spInd]) > float(spVal):
        secondDict = inputTree['right']
    else:
        secondDict = inputTree['left']

    if isinstance(secondDict, dict):
        classLabel = classify(secondDict, testVec, labels)
    else:
        classLabel = secondDict
    return classLabel


import time

'''划分训练数据与测试数据'''
def split_train_test(dataset, ratio=0.2):
    #ratio = 0.2  # 取百分之二十的数据当做测试数据
    num = len(dataset)
    train_num = int((1-ratio) * num)
    dataset_copy = list(dataset)
    traindata = list()
    while len(traindata) < train_num:
        index = randint(0,len(dataset_copy)-1)
        traindata.append(dataset_copy.pop(index))
    testdata = dataset_copy
    return traindata, testdata

if __name__ == '__main__':
    dataSet = pd.read_csv(r'Data/forestfires.csv')
    dataSet = dataSet.drop(['month', 'day'], 1)
    # dataSet = pd.read_csv('Data/irisdata.csv')
    raw, column = dataSet.shape
    col = dataSet.columns.values.tolist()
    num = dataSet.isnull().sum().sort_values()
    dataSet = dataSet.values.tolist()
    traindata, testdata = split_train_test(dataSet, ratio=0.2)
    dataSet = np.mat(dataSet)
    tree = createTree(dataSet, col)  # 输出决策树模型结果
    cor = 0
    for data in testdata:
        prediction = classify(tree, data, col)
        print(prediction, data[-1])
        cor = (prediction - data[-1])**2
    cor = cor/(len(testdata))
    print(cor)