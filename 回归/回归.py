# -*- coding:utf-8 -*-
import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')   #忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def loadDataSet(filename):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
    """
    #打开文件，以空格为分隔符
    dataSet = pd.read_csv(filename, delim_whitespace=True )
    #给每一列的数据加上一个标签，方便提取
    houseData = pd.DataFrame(data=np.array(dataSet), columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'], index=range(505))
    #提取出自住房的平均房价的影响价格
    influencingFactor = houseData[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
    #提取出自主方的平均房价
    AverageHousePrice = houseData['MEDV']
    #将两个数据集变成列表
    influencingFactor = influencingFactor.values
    influencingFactor = influencingFactor.tolist()
    AverageHousePrice = AverageHousePrice.values
    AverageHousePrice = AverageHousePrice.tolist()
    #返回数据集
    return influencingFactor, AverageHousePrice, houseData


def regularize(xMat, yMat):
    """
    函数说明:数据标准化
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集
    """
    inxMat = xMat.copy()           #数据拷贝
    inyMat = yMat.copy()
    yMean = np.mean(yMat, 0)          #行与行操作，求均值
    inyMat = yMat - yMean                 #数据减去均值
    inMeans = np.mean(inxMat, 0)           #行与行操作，求均值
    inVar = np.var(inxMat, 0)                #行与行操作，求方差
    inxMat = (inxMat - inMeans) / inVar         #数据减去均值除以方差实现标准化
    return inxMat, inyMat

def calcMean(x,y):
    """
    函数说明：得到x和y的均值，提供计算person系数
    :param x: x向量
    :param y: yxl
    :return: x_mean - x均值后向量， y_mean-y均值后的向量
    """
    sum_x = sum(x)  #求和
    sum_y = sum(y)    #求和
    n = len(x)       #得到长度
    x_mean = float(sum_x+0.0)/n   #相除
    y_mean = float(sum_y+0.0)/n   #相除
    return x_mean,y_mean

def calcPearson(x,y):
    """
    函数说明：计算Person系数
    :param x: x值
    :param y: y值
    :return: p-person系数
    """
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0   #初始化变量
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):            #代入公式
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)  #分子
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)   #分母
    p = sumTop/sumBottom
    return p


def correlateAnalysis(influencingFactor, AverageHousePrice, houseData):
    """
    函数说明：对每个特征进行相关性分析
    :param influencingFactor: 影响房屋价格的因素
    :param AverageHousePrice: 平均房价
    :param houseData: 房子的数据
    :return:
    """
    # for column in houseData:
        # p = calcPearson(houseData[column], houseData['MEDV'])
        # print(column, "和MEDV的相关系数为", p)
        # sns.jointplot(x=column, y='MEDV', data=houseData)
        # plt.show()
    # plt.rcParams['figure.figsize'] = (15, 10)
    # corrmatrix = houseData.corr()
    # sns.heatmap(corrmatrix, square=True, vmax=1, vmin=-1, center=0.0, cmap='rainbow')
    # plt.show()
    #删掉相关系数不大的特征'CHAS','DIS','B'
    influencingFactor = houseData[['CRIM','INDUS','NOX', 'RM', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']]
    influencingFactor = influencingFactor.values
    influencingFactor = influencingFactor.tolist()
    return influencingFactor, AverageHousePrice

def ProPressData(houseData):
    """
    函数说明：数据空值检查
    :param houseData: 房屋数据
    :return: none
    """
    total = houseData.isnull().sum().sort_values(ascending=False)
    precent = (houseData.isnull().sum()/houseData.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, precent], axis=1, keys=['Total', 'Precent'])
    print(missing_data)

def standRegres(xArr,yArr):
    """
    函数说明:计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k = 1.0):
    """
    函数说明:使用局部加权线性回归计算回归系数w
    Parameters:
        testPoint - 测试样本点
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))                                        #创建权重对角矩阵
    for j in range(m):                                                  #遍历数据集计算每个样本的权重
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆Erroor")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))                            #计算回归系数
    return testPoint * ws

def ridgeRegres(xMat, yMat, lam = 0.2):
    """
    函数说明:岭回归
    Parameters:
        xMat - x数据集
        yMat - y数据集
        lam - 缩减系数
    Returns:
        ws - 回归系数
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws



def crossValidation(xArr, yArr, numVal=10):
    """
    函数说明:交叉验证岭回归
    Parameters:
        xArr - x数据集
        yArr - y数据集
        numVal - 交叉验证次数
    Returns:
        wMat - 回归系数矩阵
    """
    m = len(yArr)  # 统计样本个数
    indexList = list(range(m))  # 生成索引值列表
    errorMat = np.zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):  # 交叉验证numVal次
        trainX = [];trainY = []  # 训练集
        testX = [];testY = []  # 测试集
        random.shuffle(indexList)  # 打乱次序
        for j in range(m):  # 划分数据集:90%训练集，10%测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # 获得30个不同lambda下的岭回归系数
        for k in range(30):  # 遍历所有的岭回归系数
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)  # 测试集
            meanTrain = np.mean(matTrainX, 0)  # 测试集均值
            varTrain = np.var(matTrainX, 0)  # 测试集方差
            matTestX = (matTestX - meanTrain) / varTrain  # 测试集标准化
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # 根据ws预测y值
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))  # 统计误差
    meanErrors = np.mean(errorMat, 0)  # 计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))  # 找到最小误差
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]  # 找到最佳回归系数
    xMat = np.mat(xArr);yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0);varX = np.var(xMat, 0)
    unReg = bestWeights / varX  # 数据经过标准化，因此需要还原
    unReg = np.array(unReg)
    m, n = unReg.shape
    unReg = unReg.reshape(n, 1)
    unReg = np.mat(unReg)
    return unReg


def ridgeTest(xArr, yArr):
    """
    函数说明:岭回归测试
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        wMat - 回归系数矩阵
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat, axis = 0)                        #行与行操作，求均值
    yMat = yMat - yMean                                    #数据减去均值
    xMeans = np.mean(xMat, axis = 0)                    #行与行操作，求均值
    xVar = np.var(xMat, axis = 0)                        #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar                        #数据减去均值除以方差实现标准化
    numTestPts = 30                                        #30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))    #初始回归系数矩阵
    for i in range(numTestPts):                            #改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))    #lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T                                 #计算回归系数矩阵
    return wMat

def rssError(yArr, yHatArr):
    """
    误差大小评价函数
    Parameters:
        yArr - 真实数据
        yHatArr - 预测数据
    Returns:
        误差大小
    """
    return ((yArr - yHatArr) **2).sum()

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    """
    函数说明:前向逐步线性回归
    Parameters:
        xArr - x输入数据
        yArr - y预测数据
        eps - 每次迭代需要调整的步长
        numIt - 迭代次数
    Returns:
        wsMat - 回归系数矩阵
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T                                         #数据集
    xMat, yMat = regularize(xMat, yMat)                                                #数据标准化
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))                                                #初始化numIt次迭代的回归系数矩阵
    ws = np.zeros((n, 1))                                                            #初始化回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):                                                            #迭代numIt次
        # print(ws.T)                                                                    #打印当前回归系数矩阵
        lowestError = float('inf')                                          #正无穷
        for j in range(n):                                                            #遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign                                                #微调回归系数
                yTest = xMat * wsTest                                                #计算预测值
                rssE = rssError(yMat.A, yTest.A)                                    #计算平方误差
                if rssE < lowestError:                                                #如果误差更小，则更新当前的最佳回归系数
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T #记录numIt次迭代的回归系数矩阵
    wsMax = np.mat(wsMax)
    return wsMax

def pca(dataMat, topNfeat=9999999):
    """
    函数说明：数据降维
    :param dataMat: 需要降维的数据
    :param topNfeat:
    :return:
    """
    # 均值归一化
    meanVals = dataMat.mean(axis=0)
    maxVals = dataMat.max(axis=0)
    minVals = dataMat.min(axis=0)
    meanRemoved = (dataMat - meanVals) / (maxVals - minVals)
    # 协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    # 特征值，特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 按照特征值从小到大排序，返回排序后索引
    eigValInd = np.argsort(eigVals)
    # 逆序取topNfeat个最大的特征的索引
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    # 获取特征向量
    redEigVects = eigVects[:, eigValInd]
    # 矩阵相乘，降低维度
    lowDDataMat = meanRemoved * redEigVects
    # 将原始数据重新映射会高维，用于调试
    reconMat = np.multiply((lowDDataMat * redEigVects.T), (maxVals - minVals)) + meanVals
    return lowDDataMat, reconMat

def testAlgrim():
    #文件渎职
    filename = 'housing.txt'
    #得到影响房屋的因素，平均房价，房屋信息
    influencingFactor, AverageHousePrice, houseData = loadDataSet(filename)
    #对其进行相关性分析
    influencingFactor, AverageHousePrice = correlateAnalysis(influencingFactor, AverageHousePrice, houseData)
    #数据规范化
    normInfluencingFactor, AverageHousePrice = regularize(influencingFactor, AverageHousePrice)
    #数据降维
    lowerData = pca(np.array(influencingFactor))
    lowerData = np.array(lowerData)
    lowerData = np.reshape(lowerData,(-1,3))  #将降维后的数据转换成二维数组
    #分别得到未降维前岭回归后的系数，交叉回归后的系数，普通回归后的系数
    wMat = ridgeRegres(np.mat(normInfluencingFactor[0:251]), np.mat(AverageHousePrice[0:251]).T)
    unReg = crossValidation(normInfluencingFactor[0:251], AverageHousePrice[0:251], numVal=10)
    ws = standRegres(normInfluencingFactor[0:251], AverageHousePrice[0:251])
    returnMat = stageWise(normInfluencingFactor[0:251], AverageHousePrice[0:251])
    returnMat01 = stageWise(normInfluencingFactor[0:251], AverageHousePrice[0:251], 0.20, 20)
    #分别得到降维后岭回归后的系数，交叉回归后的系数，普通回归后的系数
    wMat01 = ridgeRegres(np.mat(lowerData[0:251]), np.mat(AverageHousePrice[0:251]).T)
    unReg01 = crossValidation(lowerData[0:251], AverageHousePrice[0:251], numVal=10)
    ws01 = standRegres(lowerData[0:251], AverageHousePrice[0:251])
    #统计出行数
    m = np.shape(normInfluencingFactor[252:504])[0]
    yHat = np.zeros(m); yHat01 = np.zeros(m); yHat02 = np.zeros(m)
    yHat03 = np.zeros(m); yHat04 = np.zeros(m); yHat05 = np.zeros(m)
    yHat06 = np.zeros(m); yHat07 = np.zeros(m)
    #分别得到每个系数得到的预测值
    for i in range(m):
        yHat[i] = normInfluencingFactor[252 + i] * ws
        yHat01[i] = normInfluencingFactor[252 + i] * unReg
        yHat02[i] = normInfluencingFactor[252 + i] * wMat
        yHat06[i] = normInfluencingFactor[252 + i] * returnMat
        yHat07[i] = normInfluencingFactor[252 + i] * returnMat01
        yHat03[i] = lowerData[252 + i] * ws01
        yHat04[i] = lowerData[252 + i] * unReg01
        yHat05[i] = lowerData[252 + i] * wMat01
    #输出误差大小
    print('简单的线性回归误差大小:', rssError(AverageHousePrice[252:504], yHat))
    print('经过10次交叉验证的岭回归误差大小：', rssError(AverageHousePrice[252:504], yHat01))
    print('不用交叉验证的岭回归误差大小：', rssError(AverageHousePrice[252:504], yHat02))
    print("步长为0.01,迭代次数为100次的逐步线性回归的误差大小：", rssError(AverageHousePrice[252:504], yHat06))
    print("步长为0.01， 迭代次数为20次的逐步线性回归的误差大小：", rssError(AverageHousePrice[252:504], yHat07))
    print('降维后简单的线性回归误差大小:', rssError(AverageHousePrice[252:504], yHat03))
    print('降维后经过10次交叉验证的岭回归误差大小：', rssError(AverageHousePrice[252:504], yHat04))
    print('降维后不用交叉验证的岭回归误差大小：', rssError(AverageHousePrice[252:504], yHat05))


if __name__ == "__main__":
    testAlgrim()