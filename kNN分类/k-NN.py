# -*- coding:utf-8 -*-
import warnings
import operator
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')   #忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
    attributes = iris_data[['sl', 'sw', 'pl', 'pw']]   #分离出花的属性
    iris_data['type'] = iris_data['type'].apply(lambda x: x.split('-')[1])  # 最后类别一列，感觉前面的'Iris-'有点多余即把class这一列的数据按'-'进行切分取切分后的第二个数据
    labels = iris_data['type']     #分理出花的类别
    attriLabels = []      #建立一个标签列表
    for label in labels:        #为了更方便操作，将三中不同的类型分别设为1，2，3
        if label == 'setosa':    #如果类别为setosa的话，设为1
            attriLabels.append(1)
        elif label == 'versicolor':  #如果是versicolor的时候设为2
            attriLabels.append(2)
        elif label == 'virginica':  #如果是virginica的时候设为3
            attriLabels.append(3)
    return attributes, attriLabels

def autoNorm(attributes):
    """
    函数说明： 对数据进行归一化
    :parameter
            attributes - 特征矩阵
    :return:  nonormAttributes - 归一化后的矩阵
    """
    attributes = attributes.values  #将DataFrame类型转变为array类型
    minVal = attributes.min()      #找出数据中的最小值
    maxVal = attributes.max()      #找出数据中的最大值
    ranges = maxVal - minVal        #数据范围
    normAttributes = np.zeros(np.shape(attributes))  #初始化归一化数据
    m = attributes.shape[0]     #获取数据的行数
    normAttributes = attributes - np.tile(minVal, (m, 1))  #创建一个全是最小值得数组
    normAttributes = normAttributes / np.tile(ranges, (m, 1))  #创建一个全是范围值得数组
    return normAttributes, ranges, minVal   #返回归一化后的数据

def classify0(inX, dataSet, labels, k):
    """
    函数说明：kNN算法分类器
    :param inX: 用于分类的数据（测试集）
    :param dataSet: 用于训练的数据（训练集）
    :param labels:   花的种类（分类标签）
    :param k:   选取距离最小的k个点
    :return: sortedClassCount[0][0] - 分类结果
    """
    #返回数据的行数
    dataSetSize = dataSet.shape[0]
    #将输入的测试集在行向方向复制一次，列向方向复制dataSetSize次，再相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #将相减后的数据平方
    sqDiffMat = diffMat**2
    #将平方后的数据横向相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开平方，得到测试集到每个训练集的距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #计数类别次数的字典
    classCount = {}
    for i in range(k): #遍历循环取出k个训练集
        #取出前k个标签
        voteIlabel = labels[sortedDistIndices[i]]
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #将值按字典的值进行降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回次数最多的类别
    return sortedClassCount[0][0]

def classifyPerson():
    """
    函数说明：通过输入花的四个特征属性，进行分类输出
    :return:
    """
    #输出结果列表
    resultList = ['setosa', 'versicolor', 'virginica']
    #输入四个特征
    sl = float(input("请输入花萼长度："))
    sw = float(input("请输入花萼宽度："))
    pl = float(input("请输入花瓣长度："))
    pw = float(input("请输入花瓣宽度："))
    filename = "iris.data"
    #打开文件，得到特征数组和标签数组
    attributes, labels = loadDataSet(filename)
    #对特征变量进行归一化处理，得到归一化后的数据，和数据范围和最小值
    norAttributes, range, minVal = autoNorm(attributes)
    #生成训练集
    inArr = np.array([sl, sw, pl, pw])
    #测试集归一化
    norminArr = (inArr - minVal) / range
    #得到分类结果
    classifierReslut = classify0(norminArr, norAttributes, labels, 7)
    print("该花的类型可能为%s" % (resultList[classifierReslut-1]))

def datingClassTest():
    """
    函数说明：分类器测试器
    :return: none
    """
    #打开文件名
    # filename = "iris.data"
    filename = "bezdekIris.data"
    #打开文件，得到特征数组，和标签数组
    attributes, labels = loadDataSet(filename)
    #输出每个类别的个数
    # for i in set(labels):
    #     print(i, "的个数为", labels.count(i))
    #取数据的程度
    hoRatio = 0.3
    #数据归一化，得到归一化后的矩阵，数据范围，数据最小值
    norAttributes, ranges, minVal = autoNorm(attributes)
    #由于attributes里面的数据集中类别排序太正规了，对测试算法具有较大的影响，所以随机打乱数据的下标排序
    norAttributes_indexes = np.random.permutation(len(norAttributes))
    #得到归一化后数据的行数
    m = norAttributes.shape[0]
    #取百分之hoRatio的数据
    numTestVecs = int(m * hoRatio)
    #取norAttributes的前百分之三十的下标
    test_indexes = norAttributes_indexes[:numTestVecs]
    #取后百分之70的下标
    train_indexes = norAttributes_indexes[numTestVecs:]
    #得到训练集
    x_train = np.array(norAttributes)[train_indexes]
    #得到训练集的标签
    y_train = np.array(labels)[train_indexes]
    #得到测试集
    x_test = np.array(norAttributes)[test_indexes]
    #得到测试集的标签
    y_test = np.array(labels)[test_indexes]
    #得到最优参数k
    k = adjustK(x_test, x_train, y_test, y_train, numTestVecs)
    #计算错误个数
    errorCount = 0.0
    #遍历numTestVecs次
    for i in range(numTestVecs):
        #前numTestVecs个数据为测试集，后m-numTestVecs为训练集
        classifierResult = classify0(x_test[i,:], x_train, y_train, k)
        #输出预测结果和分类结果
        print("分类结果：%d \t 真实类别：%d" % (classifierResult, y_test[i]))
        #如果不相等，就错误个数加一
        if classifierResult != y_test[i]:
            errorCount += 1.0
    #输出错误率
    print("错误率： %f%%" % (errorCount/float(numTestVecs)*100))

def adjustK(x_test, x_train, y_test, y_train, nunTestVecs):
    """
    函数说明：获取最优参数k
    :param x_test: 测试集的特征值
    :param x_train: 训练集的特征值
    :param y_test: 测试集的标签值
    :param y_train: 训练集的标签值
    :param nunTestVecs: 训练集的个数
    :return: k - 最优参数k
    """
    #定义最优的正确率
    best_correctRate = 0.0
    #最优的k值
    best_k = 0
    #让k在1到8取值
    for k in range(1,9):
        #正确的个数要放在循环内
        correctCount = 0.0
        #循环多次求最优解
        for i in range(nunTestVecs):
            #得到测试集的预测值
            y_predict = classify0(x_test[i,:], x_train, y_train, k)
            #当预测值和真实值相等的时候
            if y_predict == y_test[i]:
                correctCount += 1.0  #正确个数加一
        #求出正确率
        correctRate = correctCount / float(nunTestVecs)
        print("k=", k, "正确率为:", correctRate)
        #判断当前正确率与前面出现的正确率的大小
        if correctRate > best_correctRate:
            best_correctRate = correctRate  #将最大值赋值给最优值
            best_k = k
    print("正确率最大时k的取值为:", best_k)  #输出最优参数k
    return best_k

if __name__ == "__main__":
    time1 = time()
    # datingClassTest() #测试数据集
    classifyPerson()   # 使用数据集
    time2 = time()
    time = time2 - time1
    print(f"耗时：{time}秒")