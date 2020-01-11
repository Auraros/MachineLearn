# -*- coding:utf-8 -*-
from random import randint
from csv import reader
from random import seed
import pandas as pd
import random
import math
import operator

def classify(inputTree, featLable , testVec):
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
    m = len(label)
    bestFeatureIndex = randint(0, m-1)
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

class extraTrees(object):
    def __init__(self,trees_num,  feature_ratio,  labels):
        self.trees_num = trees_num                # 森林的树的数目
        self.feature_ratio = feature_ratio        # 特征比例（列采样）
        self.trees = list()                       # extra tree
        self.labels = labels

    def col_sample(self, dataSet1, labels, n_features):
        features_index = []
        labels_name = []
        # for n in range(n_features):
        #     seed(n)
        #     features_index.append(randint(0, n_features))
        n = len(labels)
        dataSet = [data[:-1] for data in dataSet1]
        features_index = random.sample(range(n), n_features)
        # features_index = set(features_index)
        dataSet = pd.DataFrame(dataSet)
        dataSet = dataSet[features_index].values.tolist()
        for i in features_index:
            labels_name.append(labels[i])
        for i in range(len(dataSet)):
            dataSet[i].append(dataSet1[i][-1])
        return dataSet, labels_name


    '''建立随机森林'''
    def build_randomforest(self, train):
        n_trees = self.trees_num
        sample = train
        for i in range(n_trees):
            labels = self.labels.copy()
            tree = createTree(sample, labels)
            self.trees.append(tree)
        return self.trees

    '''随机森林预测的多数表决'''
    def bagging_predict(self, onetestdata):
        # for tree in self.trees:
        #     print(tree)
        #     print(onetestdata)
        #     print(classify(tree,['sl', 'sw', 'pl', 'pw'],onetestdata ))
        #     input()
        predictions =[]
        for tree in self.trees:
            try:
                a = classify(tree,self.labels,onetestdata)
                predictions.append(a)
            except KeyError:
                predictions.append(onetestdata[-1])
            except AttributeError:
                predictions.append(onetestdata[-1])
            except ValueError:
                predictions.append(onetestdata[-1])
        # predictions = [classify(tree,['sl', 'sw', 'pl', 'pw'],onetestdata ) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    '''计算建立的森林的精确度'''
    def accuracy_metric(self, testdata):
        correct = 0
        for i in range(len(testdata)):
            predicted = self.bagging_predict(testdata[i])
            if testdata[i][-1] == predicted:
                correct += 1
        return correct / float(len(testdata)) * 100.0


# 数据处理
'''导入数据'''
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

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


# 测试
if __name__ == '__main__':
   dataSet = pd.read_csv(r'Data/adult.csv')
   dataSet =dataSet[[' workclass',' education',' marital-status',' occupation',' relationship',' native-country',' sex',' race', ' year-income']]
   # dataSet = dataSet.drop(['age', ''], 1)
   raw, column = dataSet.shape
   col = dataSet.columns.values.tolist()
   num = dataSet.isnull().sum().sort_values()
   dataSet = dataSet.values.tolist()
   traindata,testdata = split_train_test(dataSet, ratio=0.2)
   trees_num = 1
   feature_ratio=0.3
   myRF = extraTrees(trees_num,  feature_ratio, col[:-1])
   myRF.build_randomforest(traindata)
   acc = myRF.accuracy_metric(testdata[:-1])
   print('模型准确率：',acc,'%')