from math import log
import operator
import sys
import numpy as np
import pandas as pd
sys.path.append(r'D:\python\QG\machine-learning\decide tree\plotTree')


def calcProbabilityEnt(dataSet):
    """
    样本点属于第1个类的概率p，即计算2p(1-p)中的p
    @param dataSet: 数据集
    @return probabilityEnt: 数据集的概率
        """
    numEntries = len(dataSet)  # 数据条数
    feaCounts = 0
    fea1 = dataSet[0][len(dataSet[0]) - 1]
    for featVec in dataSet:  # 每行数据
        if featVec[-1] == fea1:
            feaCounts += 1
    probabilityEnt = float(feaCounts) / numEntries
    return probabilityEnt


def splitDataSet(dataSet, index, value):
    """
    划分数据集，提取含有某个特征的某个属性的所有数据
    @param dataSet: 数据集
    @param index: 属性值所对应的特征列
    @param value: 某个属性值
    @return retDataSet: 含有某个特征的某个属性的数据集
    """
    retDataSet = []
    feature_index = []
    for featVec in dataSet:
        # 如果该样本该特征的属性值等于传入的属性值，则去掉该属性然后放入数据集中
        if featVec[index] == value:
            feature_index.append(dataSet.index(featVec))
            reducedFeatVec = featVec[:index] + featVec[index + 1:]  # 去掉该属性的当前样本
            retDataSet.append(reducedFeatVec)  # append向末尾追加一个新元素，新元素在元素中格式不变，如数组作为一个值在元素中存在
    return retDataSet, feature_index

def splitDataSet1(dataSet, index, value):
    """
    划分数据集，提取含有某个特征的某个属性的所有数据
    @param dataSet: 数据集
    @param index: 属性值所对应的特征列
    @param value: 某个属性值
    @return retDataSet: 含有某个特征的某个属性的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        # 如果该样本该特征的属性值等于传入的属性值，则去掉该属性然后放入数据集中
        if featVec[index] == value:
            reducedFeatVec = featVec[:index] + featVec[index + 1:]  # 去掉该属性的当前样本
            retDataSet.append(reducedFeatVec)  # append向末尾追加一个新元素，新元素在元素中格式不变，如数组作为一个值在元素中存在
    return retDataSet

def chooseBestFeatureToSplit(dataSet, D):
    """
    选择最优特征
    @param dataSet: 数据集
    @return bestFeature: 最优特征所在列
    """

    numFeatures = len(dataSet[0]) - 1  # 特征总数
    if numFeatures == 1:  # 当只有一个特征时
        return 0
    bestGini = 1  # 最佳基尼系数
    bestFeature = -1  # 最优特征
    for i in range(numFeatures):
        uniqueVals = set(example[i] for example in dataSet)  # 去重，每个属性值唯一
        feaGini = 0  # 定义特征的值的基尼系数
        # 依次计算每个特征的值的熵
        for value in uniqueVals:
            subDataSet, future_index = splitDataSet(dataSet, i, value)  # 根据该特征属性值分的类
            # 参数：原数据、循环次数(当前属性值所在列)、当前属性值
            D_sum = 0
            for index in future_index:   #计算权重比
                D_sum += D[index][0]
            prob = D_sum
            # prob = len(subDataSet) / float(len(dataSet))
            fea1 = subDataSet[0][len(subDataSet[0]) - 1]
            n = len(subDataSet[0])
            Datalass, D_index = splitDataSet(subDataSet, n-1, fea1)
            probabilityEnt = 0
            for D_num in D_index:   #计算出现的概率
                probabilityEnt += D[D_num][0]/D_sum
            # probabilityEnt = calcProbabilityEnt(subDataSet)
            feaGini += prob * (2 * probabilityEnt * (1 - probabilityEnt))
        if (feaGini < bestGini):  # 基尼系数越小越好
            bestGini = feaGini
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    对最后一个特征分类，出现次数最多的类即为该属性类别，比如：最后分类为2男1女，则判定为男
    @param classList: 数据集，也是类别集
    @return sortedClassCount[0][0]: 该属性的类别
    """
    classCount = {}
    # 计算每个类别出现次数
    for vote in classList:
        try:
            classCount[vote] += 1
        except KeyError:
            classCount[vote] = 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 出现次数最多的类别在首位
    # 对第1个参数，按照参数的第1个域来进行排序（第2个参数），然后反序（第3个参数）
    return sortedClassCount[0][0]  # 该属性的类别


def createTree(dataSet, labels, D, dictnum = 1):
    """
    对最后一个特征分类，按分类后类别数量排序，比如：最后分类为2同意1不同意，则判定为同意
    @param dataSet: 数据集
    @param labels: 特征集
    @return myTree: 决策树
    """
    classList = [example[-1] for example in dataSet]  # 获取每行数据的最后一个值，即每行数据的类别
    # 当数据集只有一个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 当数据集只剩一列（即类别），即根据最后一个特征分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    if dictnum == 5:
        return majorityCnt(classList)
    else:
        dictnum += 1
    # 其他情况
    bestFeat = chooseBestFeatureToSplit(dataSet, D)  # 选择最优特征（所在列）
    bestFeatLabel = labels[bestFeat]  # 最优特征
    del (labels[bestFeat])  # 从特征集中删除当前最优特征
    uniqueVals = set(example[bestFeat] for example in dataSet)  # 选出最优特征对应属性的唯一值
    myTree = {bestFeatLabel: {}}  #分类结果以字典形式保存
    for value in uniqueVals:
        subLabels = labels[:]  # 深拷贝，拷贝后的值与原值无关（普通复制为浅拷贝，对原值或拷贝后的值的改变互相影响
        myTree[bestFeatLabel][value] = createTree(splitDataSet1(dataSet, bestFeat, value), subLabels, D, dictnum)  # 递归调用创建决策树
    return myTree

def classify(inputTree, featLable , testVec):
    """
    获取分类的结果（算法的使用器）
    :param inputTree:决策树字典
    :param featLable: 标签列表
    :param testVec: 测试向量
    :return:
    """
    #获取根节点的名称，并且把根节点变成列表
    # print(inputTree)
    firstSide = list(inputTree.keys())
    # print(firstSide)
    #根节点名称string类型
    firstStr = firstSide[0]
    # print(firstStr)
    #获得根节点对应得子节点
    secondDict = inputTree[firstStr]
    # print(secondDict)
    #获得子节点在标签列表得索引
    featIndex = featLable.index(firstStr)
    # print(featIndex)
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

def CreateTreeNation(dataSet, labels, D):
    labels1 = labels[:]
    tree = createTree(dataSet, labels1, D)
    predictedVals = [];errArr = []
    minError = float('inf')  # 最小误差初始化为正无穷大
    for data in dataSet:
        prediction = classify(tree, labels, data)
        if prediction == ' >50K':
            predictedVals.append([1.0])
        else:
            predictedVals.append([-1.0])
        # predictedVals.append(prediction)
        if prediction == data[-1]:  #正确的化成0
            errArr.append(0)
        else:   #错误率改成1
            errArr.append(1)
    errArr = np.mat(errArr)
    predictedVals = np.mat(predictedVals)
    D = np.mat(D)
    weightedError = D.T * errArr.T  # 计算误差
    D = D.tolist()
    return tree, weightedError, predictedVals
