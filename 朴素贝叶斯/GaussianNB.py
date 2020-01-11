import numpy as np
import time
from sklearn import datasets

iris = datasets.load_iris()

def getGaussianCoeff(data_X, data_y):
    values = list(set(data_y.tolist()))  # 需要分表的个数
    # print(values)
    box = []
    coeff = []
    data = np.column_stack((data_X, data_y))  # X，y连接在一起,把数据特征和标签都连接在一起
    #print(data)
    for value in values:  # 按照值进行分表
        zhongjie = []
        for i in range(len(data)):
            if data[i][-1] == value:
                zhongjie.append(data[i].tolist())
        box.append(zhongjie)
    # 对每个属性，计算高斯分布的系数
    box = np.array(box)  # 分表转成ndarray，这是多维数组(3,50,5)
    for i in range(len(values)):  # 对于每个分表
        sigema = []
        uVector = np.average(box[i], axis=0)  # 均值向量,axis表示竖着求均值  (求出均值）
        # print(uVector)
        # 因为这是一个矩阵，求出反差可以求出协方差矩阵
        uVecyorMinusAvg = np.array([(box[i][j] - uVector) for j in range(len(box[i]))])  # x-u
        uVecyorMinusAvgTranspose = uVecyorMinusAvg.T
        xieFangChaMatrix = np.dot(uVecyorMinusAvg, uVecyorMinusAvgTranspose)  # 算出协方差
        # 遍历出[1,1],[2,2]...这些位置上的数字，也就是各个sigema
        for j in range(len(xieFangChaMatrix)):
            for k in range(len(xieFangChaMatrix)): #协方差矩阵是一个对称的矩阵，而且对角线是各个维度的方差。
                if j == k:
                    sigema.append(round(xieFangChaMatrix[j][k], 3))   #保留小数点后三位数
        coeff.append(list(zip(sigema, uVector)))
    print(coeff)
    return coeff

def gaussiFunc(sigema, u, x):  # 这里的sigema是平方的形式
    outxishu = 1 / (np.sqrt(2 * np.pi * sigema))
    inxishu = -((x - u) ** 2) / (2 * sigema)
    return outxishu * np.exp(inxishu)  #高斯公式


def trainBayes(Data_X, Data_Y, preData):
    # 求出高斯函数的系数
    coeff = getGaussianCoeff(Data_X, Data_Y)  #均值和方差
    if len(Data_X) != len(Data_Y):
        raise TypeError
    if not isinstance(Data_X, list):  # 转成list
        Data_X = Data_X.tolist()
        Data_Y = Data_Y.tolist()
    numOfPreData = len(preData)  # 需要预测的个数
    values = list(set(Data_Y))  # 分表的个数
    preDataY = []
    # 计算出数据归为这个类的概率，
    counts = [0] * len(values)
    for k in range(len(Data_X)):
        for value in values:
            if Data_Y[k] == value:
                counts[values.index(value)] = counts[values.index(value)] + 1
    P1s = [counts[l] / sum(counts) for l in range(len(values))]  # 计算出P1的概率  加权值
    for i in range(numOfPreData):
        probabilities = []
        for j in range(len(values)):  # 每种结果的概率
            # 用上面的高斯分布求每个概率，然后相乘
            P2 = 1
            for m in range(len(Data_X[0])):
                myCoeff = coeff[j][m]  # 是一个tuple,第一个元素是sigema，第二个参数是平均值u
                probaOfOneAtrr = gaussiFunc(myCoeff[0], myCoeff[1], preData[i][m])  # 第i个数据，第m个属性的高斯概率值
                P2 = P2 * probaOfOneAtrr
            probabilities.append(P2 * P1s[j])  # 把所有可能的结果的概率放入
        probabilities = np.array(probabilities)
        index = np.argmax(probabilities)  # 选取那个最大的概率
        preDataY.append(index)  # 每个数据的结果写入
    return preDataY


'''
x = np.linspace(-10,10,100)#高斯函数检测
y = gaussiFunc(1,2,x)
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()           
'''
start2 = time.perf_counter()
preData = trainBayes(iris.data, iris.target, iris.data)
end2 = time.process_time()
print('我的程序运行的时间是：', end2 - start2)
print("错误率：",(iris.target != preData).sum()/ len(iris.data))

