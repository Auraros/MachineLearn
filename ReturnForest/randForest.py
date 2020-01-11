# -*- coding:utf-8 -*-

from random import seed
from random import randint
from csv import reader

from C45DecideTree import C45tree
from ID3DecideTree import ID3tree
from CARTDecideTree import CARTDecideTtree



from random import randint
from csv import reader
from random import seed
import pandas as pd
import random
import math


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

# 随机森林类
class randomForest:
    def __init__(self,trees_num, sample_ratio, feature_ratio, treeType, labels):
        self.trees_num = trees_num                # 森林的树的数目
        self.samples_split_ratio = sample_ratio   # 采样，创建子集的比例（行采样）
        self.feature_ratio = feature_ratio        # 特征比例（列采样）
        self.trees = list()                       # 森林
        self.treeType = treeType        #使用随机森林预测的树的类型
        self.labels = labels

    '''有放回的采样，创建数据子集'''
    def sample_split(self, dataset):
        sample = list()
        n_sample = round(len(dataset) * self.samples_split_ratio)
        while len(sample) < n_sample:
            index = randint(0, len(dataset) - 2)
            sample.append(dataset[index])
        return sample

    def col_sample(self, dataSet1, labels, n_features):
        print(dataSet1)
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
        n_features = int(math.log(len(self.labels), 2))
        #n_features = len(self.labels)-1
        if self.treeType == 'ID3':
            for i in range(n_trees):
                labels = self.labels.copy()
                sample = self.sample_split(train)
                sample, labels = self.col_sample(sample, labels, n_features)
                tree = ID3tree.createTree(sample, labels)
                self.trees.append(tree)
        elif self.treeType == 'C45':
            for i in range(n_trees):
                labels = self.labels.copy()
                sample = self.sample_split(train)
                sample, labels = self.col_sample(sample, labels, n_features)
                tree = C45tree.createTree(sample, labels)
                self.trees.append(tree)
        elif self.treeType == 'CART':
            for i in range(n_trees):
                labels = self.labels.copy()
                sample = self.sample_split(train)
                sample, labels = self.col_sample(sample, labels, n_features)
                print(sample, labels)
                tree = CARTDecideTtree.createTree(sample, labels)
                self.trees.append(tree)
        elif self.treeType == 'all':
            for i in range(n_trees):
                labels = self.labels.copy()
                sample = self.sample_split(train)
                randomNumber = randint(1, 3)
                sample, labels = self.col_sample(sample, labels, n_features)
                if randomNumber == 1:
                    tree = ID3tree.createTree(sample, labels)
                elif randomNumber == 2:
                    tree = C45tree.createTree(sample, labels)
                elif randomNumber == 3:
                    tree = CARTDecideTtree.createTree(sample, labels)
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
                print(tree, onetestdata)
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
            print(predicted, testdata[i][-1])
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
   # seed(1)   #每一次执行本文件时都能产生同一个随机数
    # dataset = [[1, 1, 3, 4, 'yes'],
    #            [0, 1, 3, 3, 'yes'],
    #            [1, 0, 4, 3, 'no'],
    #            [0, 1, 2, 4, 'no'],
    #            [0, 0, 3, 3, 'no'],
    #            [1, 0, 2, 3, 'yes'],
    #            [0, 1, 2, 4, 'no']]
    # labels = ['no surfacing', 'flippers', 'table', 'type']


    # fr = open('Data/lenses.txt')
    # lines = fr.readlines()
    # dataSet = []
    # for line in lines:
    #     line_list = []
    #     newlist = line.strip().split('  ')
    #     for i in range(1, 6):
    #         line_list.append(int(newlist[i]))
    #     dataSet.append(line_list)
    # labels = ['x1', 'x2', 'x3', 'x4']
    # label1 = labels.copy()
    # #traindata,testdata = split_train_test(dataSet, ratio=0.1)
    #
    # testdata = random.sample(dataSet, 8)
    # traindata = dataSet
    # sample_ratio = 1
    # trees_num = 2
    # feature_ratio=0.3
    # treetype = input("Do you want to use 'C45' or 'ID3' or 'CART' or 'all' ?")
    # while treetype != 'ID3' and treetype != 'C45' and treetype != 'CART' and treetype != 'all':
    #     treetype = input("you should answer right type.")
    # myRF = randomForest(trees_num, sample_ratio, feature_ratio, treetype, labels)
    # myRF.build_randomforest(traindata)
    # acc = myRF.accuracy_metric(testdata[:-1])
    # print('模型准确率：',acc,'%')

   dataSet = pd.read_csv(r'Data/adult.csv')
   dataSet =dataSet[[' workclass',' education',' marital-status',' occupation',' relationship',' native-country',' sex',' race', ' year-income']]
   # dataSet = dataSet.drop(['age', ''], 1)
   raw, column = dataSet.shape
   col = dataSet.columns.values.tolist()
   num = dataSet.isnull().sum().sort_values()
   dataSet = dataSet.values.tolist()
   traindata,testdata = split_train_test(dataSet, ratio=0.2)
   sample_ratio = 1
   trees_num = 1
   feature_ratio=0.3
   treetype = input("Do you want to use 'C45' or 'ID3' or 'CART' or 'all' ?")
   while treetype != 'ID3' and treetype != 'C45' and treetype != 'CART' and treetype != 'all':
       treetype = input("you should answer right type.")
   myRF = randomForest(trees_num, sample_ratio, feature_ratio, treetype, col[:-1])
   myRF.build_randomforest(traindata)
   acc = myRF.accuracy_metric(testdata[:-1])
   print('模型准确率：',acc,'%')