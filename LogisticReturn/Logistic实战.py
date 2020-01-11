# -*- coding:UTF-8 -*-
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')   #忽略警告

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化                                        #存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()

def colicTest():
    frTrain = open('horseColicTraining.txt')                                        #打开训练集
    frTest = open('horseColicTest.txt')                                                #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)        #使用改进的随即上升梯度训练
    #trainWeights = gradAscent(np.array(trainingSet), trainingLabels)   #使用上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)

def colicTest1():
    frTrain = open('horseColicTraining.txt')                                        #打开训练集
    frTest = open('horseColicTest.txt')                                                #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)        #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:,0]))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

import pandas as pd
import time

def autoNorm(attributes):
    """
    函数说明： 对数据进行归一化
    :parameter
            attributes - 特征矩阵
    :return:  nonormAttributes - 归一化后的矩阵
    """
    print(attributes.describe())
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

if __name__ == '__main__':
    dataSet = pd.read_csv(r'Data/adult.csv')
    dataSet = pd.DataFrame(dataSet)
    dataSet.drop([' workclass',' education',' marital-status',' occupation',' relationship',' native-country',' sex',' race'], axis=1, inplace=True)
    normDataSet = autoNorm(dataSet.drop([' year-income'], 1))
    dataSet = pd.concat([normDataSet, dataSet[' year-income']], axis=1)
    print(dataSet)
    dataSet = dataSet.values.tolist()
    trainingLabels = []; testingLabels = []
    trainingDataSet = [] ; testingDataSet = []
    testDataSet = random.sample(dataSet, 2000)
    for data in dataSet:
        if data[-1] == ' <=50K':
            trainingLabels.append(0.0)
        elif data[-1] == ' >50K':
            trainingLabels.append(1.0)
        else:
            trainingLabels.append(0.0)
        trainingDataSet.append(data[:-1])
    for data in testDataSet:
        if data[-1] == ' <=50K':
            testingLabels.append(0.0)
        elif data[-1] == ' >50K':
            testingLabels.append(1.0)
        else:
            testingLabels.append(0.0)
        testingDataSet.append(data[:-1])
    trainWeights = gradAscent(np.array(trainingDataSet), trainingLabels)  # 使用改进的随即上升梯度训练
    errorCount = 0
    numTestVec = 0.0
    n = len(testingLabels)
    for i in range(n):
        if int(classifyVector(np.array(testingDataSet[i]), trainWeights[:,0]))!= int(testingLabels[i]):
            errorCount += 1
    errorRate = (float(errorCount) / n) * 100  # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)
    input()
