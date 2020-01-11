# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
from Adaboost import CARTtree



import math

## adaboost自适应器
def adaBoostTrainDS1(dataSet, labels, numIt = 40):
    '''

    :param dataArr: 数据集，分类标签，迭代次数
    :param classLabel:
    :param numIt:
    :return:
    '''
    weakClassArr = []
    m = np.shape(dataSet)[0]
    D = np.mat(np.ones((m, 1)) / m)   #初始化权重,并且平分权重(对每一个数据集划分成相同的权重）
    aggClassEst = np.mat(np.zeros((m, 1)))  #权重更新
    classLabels = []
    for data in dataSet:
        if data[-1] == ' >50K':
            classLabels.append(1.0)
        else:
            classLabels.append(-1.0)
    for i in range(numIt):
        D = D.tolist()
        bestStump, error, classEst = CARTtree.CreateTreeNation(dataSet, labels, D)
        # bestStump, error, classEst = buildStump(dataArr, classLabels, D) #构建决策树
        alpha = float(0.5 * np.log((1.0 - error)/ max(error, 1e-16))) #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha   #储存弱学习法的权重
        weakClassArr.append(bestStump)  #储存决策树
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) #计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D/ D.sum()   ##根据样本权重公式，更新样本权重
        #计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst)!= np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break    #误差为0，退出循环
    return weakClassArr, aggClassEst   #得到很多棵决策树和对应的权重



if __name__ == '__main__':
    dataSet = pd.read_csv(r'Data/adult.csv')
    # dataSet = dataSet.drop(['age',], 1)
    dataSet = dataSet[
        [' workclass', ' education', ' marital-status', ' occupation', ' relationship', ' native-country', ' sex',
         ' race', ' year-income']]
    raw, column = dataSet.shape
    col = dataSet.columns.values.tolist()
    num = dataSet.isnull().sum().sort_values()
    dataSet = dataSet.values.tolist()
    dataSet = dataSet[:200]
    weakClassArr, aggClassEst = adaBoostTrainDS1(dataSet, col)  #输入数据集和标签，得到最佳的森林和相对的分类