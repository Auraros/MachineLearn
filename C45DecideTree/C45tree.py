# -*- coding: utf-8 -*-
"""
决策树C4.5的实现
"""

from math import log
import operator
import pandas as pd


def majorityCnt(classList):
    """
    找到最大频繁向量（多数表决器）
    :param classList:训练集
    :return:最大值
    """
    classCounts = {}
    for value in classList:   #遍历训练集中的每一个变量
        if (value not in classCounts.keys()):  #如果变量不在列表中
            classCounts[value] = 0    #新建一个字典的键
        classCounts[value] += 1    #数量加一
    sortedClassCount = sorted(classCounts.items(), key=operator.itemgetter(1), reverse=True)  #排序
    return sortedClassCount[0][0]   #输出第一个，即最大值

def splitDataSet(dataSet, axis, value):
    """
    以靠指定列的指定值来划分数据集，比如划分西瓜瓜皮形状为椭圆的数据集
    :param axis: 索引列，即形状
    :param value: 索引列的特定值，即椭圆
    :return:
    """
    retDataSet = []
    for festdataVal in dataSet:
        if festdataVal[axis] == value:
            reducedFeatVal = festdataVal[:axis]    #这两行去掉索引列
            reducedFeatVal.extend(festdataVal[axis+1:])
            retDataSet.append(reducedFeatVal)
    return retDataSet

def calcShannonEnt(columnIndex, dataSet):
    """
    计算香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)   #获得数据集的长度
    labelCounts = {}     #计算标签的字典
    for featDataVal in dataSet:
        currentLabels = featDataVal[columnIndex]   #取最后一个标签值
        if currentLabels not in labelCounts.keys():  #判断有没有在标签里面
            labelCounts[currentLabels] = 0
        labelCounts[currentLabels] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():   #key有几个遍历几次
        prob = labelCounts[key]/float(numEntries)  #计算频率
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def chooseBestFeatureToSplitOfFurther(dataSet):
    """
            选择信息增益率最大的特征值
            :param dataSet:数据集
            :return:
            """
    numFeatures = len(dataSet[0]) - 1  # 看数据集而定，数据中如果最后一行为标签，则删去
    baseEntropy = calcShannonEnt(-1, dataSet)  # 计算所有数据集的香农熵
    bestFeaturesindex = 0  # 最佳特征的索引
    bestInfoGainRatio = 0.0  # 最佳信息熵

    for i in range(numFeatures):  # 有几个特征值循环几次
        featEntropy = calcShannonEnt(i, dataSet)
        featList = []  # 特征值列表
        for example in dataSet:  # 获得这个列的值
            featList.append(example[i])
        uniqueVals = set(featList)  # 相同的数据并没有意义，去重
        newEntropy = 0.0  # 新信息熵
        for value in uniqueVals:  # 得到该列的特征值
            subDataSet = splitDataSet(dataSet, i, value)  # 划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 权重或者条件概率
            newEntropy += prob * calcShannonEnt(-1, subDataSet)  # 计算信息增益后面的条件经验熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if featEntropy == 0.0 :
            infoGainRatio = 0.0
        else:
            infoGainRatio = infoGain/float(featEntropy)
        if infoGainRatio > bestInfoGainRatio:  # 更改最大经验熵
            bestInfoGainRatio = infoGainRatio
            bestFeaturesindex = i
    return bestFeaturesindex  # 输出最大经验熵的索引

def diferFeature(dataSet, label):
    numFeatures = len(dataSet[0]) - 1  # 看数据集而定，数据中如果最后一行为标签，则删去
    baseEntropy = calcShannonEnt(-1, dataSet)  # 计算所有数据集的香农熵
    sumEntropy = 0.0
    featureEntropy = []
    dellabel = []
    retDataSet = dataSet
    label1 = label.copy()
    for i in range(numFeatures):  # 有几个特征值循环几次
        featList = []  # 特征值列表
        for example in dataSet:  # 获得这个列的值
            featList.append(example[i])
        uniqueVals = set(featList)  # 相同的数据并没有意义，去重
        newEntropy = 0.0  # 新信息熵
        for value in uniqueVals:  # 得到该列的特征值
            subDataSet = splitDataSet(dataSet, i, value)  # 划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 权重或者条件概率
            newEntropy += prob * calcShannonEnt(-1, subDataSet)  # 计算信息增益后面的条件经验熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        featureEntropy.append(infoGain)
        sumEntropy += infoGain
    averageEntropy = sumEntropy/numFeatures
    for i in range(numFeatures):
        if featureEntropy[i]<averageEntropy:
            dellabel.append(labels[i])
    label.append('jieguo')
    for i in range(len(dellabel)):
        label1.remove(dellabel[i])
    retDataSet = pd.DataFrame(retDataSet, columns=label)
    retDataSet = retDataSet.drop(dellabel, axis=1)
    retDataSet = retDataSet.values.tolist()
    return retDataSet, label1


def createTree(dataSet, label):
    """
    创建树
    :param dataSet:
    :param label:
    :return:
    """
    classList = [] #获得每一个标签
    for classVal in dataSet:
        classList.append(classVal[-1])
    if classList.count(classList[0]) == len(classList): #如果全部标签都相同
        return classList[0]  #返回该标签
    if len(dataSet[0]) == 1:  #如果一列只有一个特征
        return majorityCnt(classList)
    #dataSet, label = self.diferFeature(dataSet, label)
    #获取最优的索引值
    bestFeatureIndex = chooseBestFeatureToSplitOfFurther(dataSet)
    #获取最优索引值的名称
    bestFeatureLabel = label[bestFeatureIndex]
    mytree = {bestFeatureLabel:{}}  #创建根节点
    del(label[bestFeatureIndex])   #删去用过的最优节点
    bestFeature = []   #最优的特征
    for example in dataSet:
        bestFeature.append(example[bestFeatureIndex])
    uniquesVal = set(bestFeature)  #最优特征的种类
    for val in uniquesVal:
        subLabel = label[:]  #创建个子标签
        mytree[bestFeatureLabel][val] = createTree(splitDataSet(dataSet, bestFeatureIndex, val), subLabel)  #递归
    return mytree

def classify( inputTree, featLable, testVec):
    """
    获取分类的结果（算法的使用器）
    :param inputTree:决策树字典
    :param featLable: 标签列表
    :param testVec: 测试向量
    :return:
    """
    #获取根节点的名称，并且把根节点变成列表
    firstSide = list(inputTree.keys())
    #根节点名称string类型
    firstStr = firstSide[0]
    #获得根节点对应得子节点
    secondDict = inputTree[firstStr]
    #获得子节点在标签列表得索引
    featIndex = featLable.index(firstStr)
    #获得测试向量的值
    key = testVec[featIndex]
    #获取树干向量后的变量
    valueOfFeat = secondDict[key]
    #判断是子结点还是叶子节点：子结点就回调分类函数，叶子结点就是分类结果
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLable, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    #写入文件
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    #读取数
    import  pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

import random
import time
from DecideTreePlanting import plotTrees

if __name__ == "__main__":
    fr = open('Data/lenses.txt')
    lines = fr.readlines()
    dataSet = []
    for line in lines:
        line_list = []
        newlist = line.strip().split('  ')
        for i in range(1, 6):
            line_list.append(int(newlist[i]))
        dataSet.append(line_list)
    labels = ['x1', 'x2', 'x3', 'x4']
    label1 = labels.copy()

    testDataSet = random.sample(dataSet, 4)
    index = []
    for test in testDataSet:
        index.append(dataSet.index(test))
    index.sort(reverse=True)
    for i in index:
        del dataSet[i]
    time1 = time.clock()
    mytree = createTree(dataSet, label1)
    time2 = time.clock()
    print(time2-time1)
    plotTrees.createPlot(mytree)

    prediction = []
    test_features = []
    test_target = []
    for testData in testDataSet:
        test_features.append(testData[:-1])
        test_target.append(testData[-1])
    for features in test_features:
        prediction.append(classify(mytree, labels, features))
    correct = [1 if a == b else 0 for a, b in zip(prediction, test_target)]
    print(correct.count(1) / len(correct))  # 计算准确率
