import numpy as np
import matplotlib.pyplot as plt


def gradAscent(dataSet, labelSet):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelSet).T
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weight = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weight)
        error = labelMatrix - h
        weight = weight + alpha * dataMatrix.T * error
    return weight.getA()


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # 以空格切分数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 添加数据集
        labelMat.append(int(lineArr[2]))  # 添加标签集
    fr.close()  # 关闭文件
    return dataMat, labelMat


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()



if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)







def plotData():
    dataMat, labelMat = loadDataSet()   #数据集和标签集
    dataArr = np.array(dataMat)          #将数据集转化为数组
    n = len(dataArr)    #数据的个数
    right_x = []; right_y=[]   #正确数据的x值和y值
    wrong_x = []; wrong_y=[]   #错误数据的x值和y值
    for i in range(n):     #循环遍历
        if labelMat[i] == 1:     #如果是正确饿值
            right_x.append(dataArr[i][1])   #保存数据
            right_y.append(dataArr[i][2])
        else:
            wrong_x.append(dataArr[i][1])
            wrong_y.append(dataArr[i][2])
    plt.scatter(right_x, right_y, s=20, c='red', marker='s', alpha=.5, label='right')  # 绘制正样本   #画正确的图
    plt.scatter(wrong_x, wrong_y, s=20, c='green', alpha=.5, label='wrong')  # 绘制负样本   #画错误的图
    plt.title('DataSet')  # 绘制title
    plt.xlabel('x')
    plt.ylabel('y')  # 绘制label
    plt.legend(loc='lower left')
    plt.show()