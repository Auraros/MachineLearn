# -*- coding:utf-8 -*-

from random import seed
from random import randint
from csv import reader
from CARTReturnTree import CARTRturnTree

from random import randint
from csv import reader
from random import seed
import numpy as np
import random
import pandas as pd

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



# ewai森林类
class extraReturnForest:
    def __init__(self,trees_num, sample_ratio, feature_ratio, labels):
        self.trees_num = trees_num                # 森林的树的数目
        self.samples_split_ratio = sample_ratio   # 采样，创建子集的比例（行采样）
        self.feature_ratio = feature_ratio        # 特征比例（列采样）
        self.trees = list()                       # 森林
        self.labels = labels

    def col_sample(self, dataSet1, labels, n_features):
        features_index = []
        labels_name = []
        # for n in range(n_features):
        #     seed(n)
        #     features_index.append(randint(0, n_features))
        n = len(labels) - 1
        dataSet = [data[:-1] for data in dataSet1]
        features_index = random.sample(range(n), n_features)
        # features_index = set(features_index)
        dataSet = pd.DataFrame(dataSet)
        dataSet = dataSet[features_index].values.tolist()
        for i in features_index:
            labels_name.append(labels[i])
        for i in range(len(dataSet)):
            dataSet[i].append(dataSet1[i][-1])
        labels_name.append(labels[-1])
        return dataSet, labels_name


    '''有放回的采样，创建数据子集'''
    def sample_split(self, dataset):
        sample = list()
        n_sample = round(len(dataset) * self.samples_split_ratio)
        while len(sample) < n_sample:
            index = randint(0, len(dataset) - 2)
            sample.append(dataset[index])
        return sample

    '''建立随机森林'''
    def build_randomforest(self, train):
        n_trees = self.trees_num
        n_features = int(self.feature_ratio * (len(train[0])-1))#列采样，从M个feature中，选择m个(m<<M)
        for i in range(n_trees):
            # sample = self.sample_split(train)
            labels = self.labels.copy()
            sample, labels = self.col_sample(train, labels, 2)
            sample = np.mat(sample)
            tree = CARTRturnTree.createTree1(sample, labels)
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
                labels = self.labels.copy()
                a = classify(tree, onetestdata, labels)
                predictions.append(a)
            except KeyError:
                continue
            except AttributeError:
                continue
        # predictions = [classify(tree,['sl', 'sw', 'pl', 'pw'],onetestdata ) for tree in self.trees]
        sum1 = 0
        for i in range(len(predictions)):
            sum1 += predictions[i]
        meanVal = sum1/len(predictions)
        return meanVal


    '''计算建立的森林的误差平方和'''

    def accuracy_metric1(self, testdata):
        sum1 =0
        for i in range(len(testdata)):
            predicted = self.bagging_predict(testdata[i])
            sum1 += (testdata[i][-1] - predicted)**2
        return sum1

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


def autoNorm(attributes):
    """
    函数说明： 对数据进行归一化
    :parameter
            attributes - 特征矩阵
    :return:  nonormAttributes - 归一化后的矩阵
    """
    attributes1 = attributes.values  #将DataFrame类型转变为array类型
    minVal = attributes1.min()      #找出数据中的最小值
    maxVal = attributes1.max()      #找出数据中的最大值
    ranges = maxVal - minVal        #数据范围
    normAttributes = np.zeros(np.shape(attributes1))  #初始化归一化数据
    m = attributes1.shape[0]     #获取数据的行数
    normAttributes = attributes1 - np.tile(minVal, (m, 1))  #创建一个全是最小值得数组
    normAttributes = normAttributes / np.tile(ranges, (m, 1))  #创建一个全是范围值得数组
    normAttributes = pd.DataFrame(normAttributes)
    return normAttributes   #返回归一化后的数据

import  time
# 测试
if __name__ == '__main__':

    dataSet = pd.read_csv(r'Data/forestfires.csv')
    dataSet = dataSet.drop(['month', 'day'], 1)
    # normDataSet = autoNorm(dataSet.drop(['area'], 1))
    # dataSet = pd.concat([normDataSet, dataSet['area']], axis=1)
    normDataSet = autoNorm(dataSet)
    raw, column = dataSet.shape
    col = dataSet.columns.values.tolist()
    num = dataSet.isnull().sum().sort_values()
    dataSet = dataSet.values.tolist()
    time1 = time.clock()

    traindata, testdata = split_train_test(dataSet, ratio=0.2)
    length = int(len(dataSet) * 0.2)
    sample_ratio = 1
    trees_num = 200
    feature_ratio = 0.3
    labels = col
    myRF = extraReturnForest(trees_num, sample_ratio, feature_ratio, labels)
    myRF.build_randomforest(traindata)
    sse = myRF.accuracy_metric1(testdata)
    print('误差比：', sse / length)