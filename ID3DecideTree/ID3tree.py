# -*- coding: utf-8 -*-
"""
决策树ID3的实现
"""

from math import log
import operator
import numpy as np
import random
import pandas as pd
from  DecideTreePlanting import plotTrees

# 输入：训练集
# D =${(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)};$
# 属性集
# A =${a_1, a_2, ..., a_d}
# 过程：函数TreeGenerate（D，A）
# 生成结点
# node；
# if D 中样本全属于同一个类别C  then
# 将node
# 标记为C类的叶节点；return
# end if
# if A = null or D中样本在A的取值相同 then
# 将node
# 标记为叶结点，其类别标记为D中样本最多的类；return
# end if
# 从A中选择最优的划分属性a *；
# for a * 的每一个值a * v   do
#     为node生成一个分支，令Dv表示D在a * 上取值为a * v的样本子集；
#     if Dv为空 then
#     将分支点标记为叶节点，其类别标记为D中样本最多的类；return；
#     else
#     以
#     TreeGenerate（Dv， A、{a *}}为分支节点
#     end if
#     end
#     for
#     输出：以node为节点的决策树

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

def calcShannonEnt(dataSet):
    """
    计算香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)   #获得数据集的长度
    labelCounts = {}     #计算标签的字典
    for featDataVal in dataSet:
        currentLabels = featDataVal[-1]   #取最后一个标签值
        if currentLabels not in labelCounts.keys():  #判断有没有在标签里面
            labelCounts[currentLabels] = 0
        labelCounts[currentLabels] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():   #key有几个遍历几次
        prob = labelCounts[key]/float(numEntries)  #计算频率
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def chooseBestFeatureToSplit(dataSet):
    """
    选择信息增益最大的特征值
    :param dataSet:数据集
    :return:
    """
    numFeatures = len(dataSet[0]) - 1 #看数据集而定，数据中如果最后一行为标签，则删去
    baseEntropy = callable(dataSet)   #计算所有数据集的香农熵
    bestFeaturesindex = 0    #最佳特征的索引
    bestEntropy = 0.0     #最佳信息熵

    for i in range(numFeatures):   #有几个特征值循环几次
        featList = []    #特征值列表
        for example in dataSet:   #获得这个列的值
            featList.append(example[i])
        uniqueVals = set(featList)  #相同的数据并没有意义，去重
        newEntropy = 0.0   #新信息熵
        for value in uniqueVals:   #得到该列的特征值
            subDataSet = splitDataSet(dataSet, i, value)  #划分数据集
            prob = len(subDataSet) / float(len(dataSet))  #权重或者条件概率
            newEntropy += prob * calcShannonEnt(subDataSet)    #计算信息增益后面的条件经验熵
        infoGain = baseEntropy - newEntropy     #计算信息增益
        if infoGain > bestEntropy:   #更改最大经验熵
            baseEntropy = infoGain
            bestFeaturesindex = i
    return bestFeaturesindex    #输出最大经验熵的索引

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
    #获取最优的索引值
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
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

def classify(inputTree, featLable, testVec):
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
    dataSet = iris_data.values.tolist()
    return dataSet

def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, 3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

import time

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
    # index = []
    # for test in testDataSet:
    #     index.append(dataSet.index(test))
    # index.sort(reverse=True)
    # for i in index:
    #     del dataSet[i]
    time1 = time.clock()
    mytree = createTree(dataSet, label1)
    time2  = time.clock()
    print(time2-time1)
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


