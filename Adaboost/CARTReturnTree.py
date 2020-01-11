
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
    return np.mean(dataSet[:, -2])


def regErr(dataSet):
    """
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差

    """
    weight = 0
    for data in dataSet.tolist():
        weight += data[-1]
    return weight*np.var(dataSet[:, -2]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet,  leafType=regLeaf,  errType=regErr, ops=(1, 4)):
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
    tolS = ops[0]
    tolN = ops[1]
    # 如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:, -2].T.tolist()[0])) == 1:  #因为加多了一个权重，所以要多一个-2
        return None, leafType(dataSet)
    # 统计数据集合的行m和列n
    m, n = np.shape(dataSet)   #计算dataSet的行列，多了一个权重，所以n要减一
    n = n - 1
    # 默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    # 遍历所有特征列
    for featIndex in range(n-1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN,则退出
            if splitVal > (dataSet[:, featIndex].max()-dataSet[:,featIndex].min())*1/2:
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
                continue
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


def createTree(dataSet, labels,  leafType=regLeaf, errType=regErr, ops=(1, 4)):
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
    # try:
    #     print(labels[feat])
    # except TypeError:
    #     pass
    if feat == None: return val
    # 回归树
    retTree = {}
    n = labels[feat]
    retTree['spInd'] = n
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet, labels,  leafType, errType, ops)
    retTree['right'] = createTree(rSet, labels,  leafType, errType, ops)
    return retTree



def CreateTreeNation(dataSet, labels):
    labels1 = labels[:]
    tree = createTree(dataSet, labels1)
    predictedVals = [];errArr = []
    maxVar = 0  # 最小误差初始化为
    for data in dataSet.tolist():
        prediction = classify(tree, data[:-1], labels)
        predictedVals.append([prediction])
        # predictedVals.append(prediction)
        var = abs(prediction - data[-2])
        if var > maxVar:
            maxVar = var
        errArr.append(var)
    for i in range(len(errArr)):
        errArr[i] = errArr[i] / maxVar
    errArr = np.mat(errArr)
    predictedVals = np.mat(predictedVals)
    D = dataSet[:, -1]
    weightedError = D.T * errArr.T  # 计算误差
    return tree, weightedError, predictedVals, errArr


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
    #获得根节点对应得子节点
    # #获得子节点在标签列表得索引
    # featIndex = testVec[secondDict]
    # print(featIndex)
    # input()
    # #获得测试向量的值
    # key = testVec[featIndex]
    # print(key)
    # input()
    # #获取树干向量后的变量
    # valueOfFeat = secondDict[key]
    #判断是子结点还是叶子节点：子结点就回调分类函数，叶子结点就是分类结果
    if isinstance(secondDict, dict):
        classLabel = classify(secondDict, testVec, labels)
    else:
        classLabel = secondDict
    return classLabel