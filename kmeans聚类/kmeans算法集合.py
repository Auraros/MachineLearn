# -*- coding:utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')   #忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#伪代码如下
# 将所有点看成一个簇
# 当簇数目小于k时
# 对于每一个簇
#     计算总误差
#     在给定的簇上面进行k-均值聚类（k=2）
#     计算将该簇一分为二之后的总误差
# 选择使得总误差最小的簇进行划分

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
    attributes = attributes.values
    return attributes, attriLabels

def randCent(dataSet, k):
    """
    函数说明：随机初始化k个质心（质心满足数据边界之内）
    :param dataSet:数据集
    :param k:  质点个数
    :return:centroids 返回初始化得到的k个质心向量
    """
    m, n = dataSet.shape  # 得到行列数目
    centroids = np.zeros((k, n))  # 定义一个k行n列的全为0的质点集合
    for j in range(n):
        #得到该列数据的最小值
        minJ = min(dataSet[:, j])
        #得到该列数据的范围
        rangeJ = float(max(dataSet[:, j]) - minJ)
        #k个质心向量的第j维数据值随机为位于范围内的某一个值
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k))
    return centroids


def distEclud(vecA,vecB):
    """
    函数说明：欧式距离公式
    :param vecA:样本点
    :param vecB：质心
    :return:距离
    """
    return np.sqrt(np.sum((vecA - vecB) ** 2))

def KMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    函数说明：k均值算法
    :param dataSet: 特征集
    :param k: k个值
    :return:centroids, clusterAssment
    """
    m = np.shape(dataSet)[0]     #得到行的数目
    clusterAssment = np.mat(np.zeros((m, 2))) #初始化一个（m，2）矩阵
    clusterChange = True      #定义聚类是否发生变化
    centroids = createCent(dataSet, k)        #创建初始化的k个质心向量
    while clusterChange:          #只要聚类结果一直发生变化，就一直执行聚类算法，直到所有数据点聚类结果不发生变化
        clusterChange = False  #聚类结果定义为False
        for i in range(m):     #遍历数据集中的每一个样本
            minDist = np.inf       #初始化最小距离为100000
            minIndex = -1    #最小距离索引定为-1
            for j in range(k): #循环k个类的质心
                distance = distMeas(centroids[j, :], dataSet[i, :]) #计算数据点到质心的欧式距离
                if distance < minDist:  #如果当前距离少于最小距离
                    minDist = distance#当前距离定为最小距离，最小距离的索引定为j
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  #当前聚类结果中第i个样本的结果发生变化
                clusterChange = True    #把clusterChange定义为Ture，代表发生了变化
            clusterAssment[i, :] = minIndex, minDist**2     #更新当前新变化的聚类结果和错误平方
        for j in range(k):                #遍历每一个质心
            #因此首先先比较clusterAssment[:,0].A==cent的真假，如果为真则记录了他所在的行，因此在用切片进行取值。
            # print(clusterAssment[:, 0].A == j)
            if (clusterAssment[:, 0].A == j).all() == False:   #再chooseK防止报错
                continue
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
    # #返回k个聚类，聚类结果和误差
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeans = distEclud):
    """
    函数说明：二分k-均值聚类算法
    :param dataSet: 待聚类的数据集
    :param k: 聚类的个数
    :param distMeans: 用户指定的距离计算方法，这里为欧式距离公式
    :return: np.mat(centList)-质心向量
            clusterAssment - 聚类结果
    """
    m = np.shape(dataSet)[0]    #获得数据集的样本数
    clusterAssment = np.mat(np.zeros((m, 2)))   #初始化一个元素全为0的（m，2）的矩阵
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]   #获取数据每一列的均值，组成一个一维数组
    centList = [centroid0]  #当前聚类列表将数据聚为一类
    for j in range(m):  #遍历数据中的每个数据集样本
        #计算当前聚类为一类时各个数据点距离质心的平方距离
        clusterAssment[j, 1] = distMeans(np.mat(centroid0), np.mat(centroid0)) ** 2
    while (len(centList) < k):   #循环，直到达到k类
        lowestSSE = np.inf    #将当前最小误差设置为正无穷大
        for i in range(len(centList)):  #遍历每个聚类
            ##因此首先先比较clusterAssment[:,0].A==cent的真假，如果为真则记录了他所在的行，因此在用切片进行取值。
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A==i)[0], :]
            # 对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat, splitClustAss = KMeans(ptsInCurrCluster, 2, distMeans)
            #计算该划分后两个类的误差平方和
            sseSplit = np.sum(splitClustAss[:, 1])
            #计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A!=i)[0], 1])
            #划分第i类后总误差小于当前最小总误差
            if(sseSplit + sseNotSplit) < lowestSSE:
                # 第i类作为本次划分类
                bestCentToSplit = i
                # 第i类划分后得到的两个质心向量
                bestNewCents = centroidMat.copy()
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss = splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowestSSE = sseSplit + sseNotSplit
        # 数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为1
        # 当前类个数+1，作为新的一个聚类
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        # 连续不出现空缺
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # # 更新质心列表中的变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0, :]
        # 添加新的类的质心向量
        centList.append(bestNewCents[1, :])
        # 更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss.copy()
    #返回聚类结果
    return np.mat(centList), clusterAssment


def chooseK(dataSet, i):
    """
    函数说明：肘部图的绘画
    :param dataSet: 数据集
    :param i:k从1开始迭代到k次
    :return:none
    """
    list = []  #定义存放距离的列表，即y轴
    x= []      #定义存放1，2，3，4，5，即x轴
    for j in range(1, i):
        #得到聚点， 聚类结果，和误差
        cenList, clusterAssment = KMeans(dataSet, j)
        #计算每个点的误差平方和，并加入到列表中
        list.append(sum(clusterAssment[:, 1]))
        #添加x轴
        x.append(j)
    #将list转变为列表
    a = np.array(list)
    list = a.reshape(-1)
    #以x为画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, list)
    plt.show()


def writeTxt(cenList1, clusterAssment, type=1):
    """
    函数说明：将质心和分类结果写入文件的操作。
    :param cenList1:质心
    :param clusterAssment:分类结果
    :param type: 1为二分k-means， 2为k-means
    :return: none
    """
    #将clusterAssment变成列表
    clusterAssment = clusterAssment.tolist()
    #获得cenList的长度
    n = len(cenList1)
    #获得clusterAssment的长度
    m = len(clusterAssment)
    #判断是否为k-means
    if type == 1:
        file = open('consequeue01.txt', mode='w')   #如果是二分k-means则打开01文件
        file.write("二分k-means的聚类后的质心")
    else:
        file = open('consequeue02.txt', mode='w')   #如果是k-means则打开02文件
        file.write("k-means的聚类后的质心")
    for j in range(n):
        file.write("\n第%d个质心为：" % (j+1))  #输入质心
        file.write(str(cenList1[j]))           #输入质心
    file.write("\n聚类结果:\n")
    for i in range(m):
        file.write('第%d个属性被归类为：' % (i+1))     #输入类别
        file.write(str(int(clusterAssment[i][0])))
        file.write("       ")
        file.write("距离为：")
        file.write(str(clusterAssment[i][1]))    #输入距离
        file.write("\n")
    file.close()


# k-means算法
# if __name__ == '__main__':
#     filename = "iris.data"      #文件路径
#     attributes, labels = loadDataSet(filename)   #得到数据
#     centList, clusterAssment = KMeans(attributes, 3)   #k=3时得到质心和分类结果
#     # chooseK(attributes, 7)     #画图
#     writeTxt(centList, clusterAssment, 2)   #写入文件

#二分k-means算法
if __name__ == '__main__':
    filename = "iris.data"      #文件路径
    attributes, labels = loadDataSet(filename)   #得到数据
    centList, clusterAssment = biKmeans(attributes, 3)   #k=3时得到质心和分类结果
    writeTxt(centList, clusterAssment, 1)   #写入文件

