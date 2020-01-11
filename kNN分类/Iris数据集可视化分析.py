# -*- coding:utf-8 -*-
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatchs

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
    labels = iris_data['type']            #分理出花的类别
    return attributes, labels

def showdatas(attributes, datinglabels):
    """
    函数说明：画出花的属性两两之间的关系图
    :parameter:
            attributes - 花的属性
            datinglabels - 花的类别
    :return:none
    """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 8))   #定义一个3行2列的画布
    LabelsColors = []    #建立一个颜色标签列表
    for i in datinglabels:   #遍历花的类型
        if i == 'setosa':    #setosa类型的花画成黑色的点
            LabelsColors.append('black')
        if i == 'versicolor':   #versicolor类型的花画成橙色的点
            LabelsColors.append('orange')
        if i == 'virginica':    #virginica类型的花画成红色的点
            LabelsColors.append('red')

    #在画板第一行第一列的位置绘制花萼长度和花萼宽度之间的关系
    axs[0][0].scatter(x=attributes['sl'], y=attributes['sw'], color=LabelsColors, s=15, alpha=.5) #x轴为花萼长度，y轴为花萼宽度， 点大小为15， 透明度为0.5
    axs0_title_text = axs[0][0].set_title(u'花萼长度和花萼宽度')     #设置title
    axs0_xlabel_text = axs[0][0].set_xlabel(u'花萼长度')            #设置x轴的标签
    axs0_ylabel_text = axs[0][0].set_ylabel(u'花萼宽度')            #设置y轴的标签
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #在画板第一行第二列的位置绘制花萼长度和花瓣长度之间的关系
    axs[0][1].scatter(x=attributes['sl'], y=attributes['pl'], color=LabelsColors, s=15, alpha=.5) #x轴为花萼长度，y轴为花瓣长度，点的大小为15， 透明度为0.5
    axs1_title_text = axs[0][1].set_title(u'花萼长度和花瓣长度')    #设立title
    axs1_xlabel_text = axs[0][1].set_xlabel(u'花萼长度')           #设置x轴标签
    axs1_ylabel_text = axs[0][1].set_ylabel(u'花瓣长度')           #设置y轴标签
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #在画板第二行第一列的位置绘制花萼长度与花瓣宽度之间的关系
    axs[1][0].scatter(x=attributes['sl'], y=attributes['pw'], color=LabelsColors, s=15, alpha=.5) #x轴为花萼长度，y轴为花瓣长度，点的大小为15， 透明度为0.5
    axs2_title_text = axs[1][0].set_title(u'花萼长度和花瓣宽度')    #设立title
    axs2_xlabel_text = axs[1][0].set_xlabel(u'花萼长度')           #设立x轴标签
    axs2_ylabel_text = axs[1][0].set_ylabel(u'花瓣宽度')          #设立y轴标签
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    #在画板第二行第二列的位置上绘制花萼宽度与花瓣长度之间的关系
    axs[1][1].scatter(x=attributes['sw'], y=attributes['pl'], color=LabelsColors, s=15, alpha=.5)  #x轴为花萼宽度，y轴为花瓣长度
    axs3_title_text = axs[1][1].set_title(u'花萼宽度和花瓣长度')    #设立title
    axs3_xlabel_text = axs[1][1].set_xlabel(u'花萼宽度')          #设立x轴标签
    axs3_ylabel_text = axs[1][1].set_ylabel(u'花瓣长度')          #设立y轴标签
    plt.setp(axs3_title_text, size=9, weight='bold', color='red')
    plt.setp(axs3_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs3_ylabel_text, size=7, weight='bold', color='black')

    #在画板第三行第一列的位置绘制花萼宽度与花瓣宽度之间的关系
    axs[2][0].scatter(x=attributes['sw'], y=attributes['pw'], color=LabelsColors, s=15, alpha=.5)  #x轴为花萼宽度，y轴为花瓣宽度
    axs4_title_text = axs[2][0].set_title(u'花萼宽度和花瓣宽度')   #设立title
    axs4_xlabel_text = axs[2][0].set_xlabel(u'花萼宽度')          #设立x轴坐标
    axs4_ylabel_text = axs[2][0].set_ylabel(u'花瓣宽度')          #设立y轴坐标
    plt.setp(axs4_title_text, size=9, weight='bold', color='red')
    plt.setp(axs4_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs4_ylabel_text, size=7, weight='bold', color='black')

    #在画板第三行第二列的位置绘制花瓣长度和花瓣宽度之间的关系
    axs[2][1].scatter(x=attributes['pl'], y=attributes['pw'], color=LabelsColors, s=15, alpha=.5)  #x轴花瓣长度，y轴为花瓣宽度
    axs5_title_text = axs[2][1].set_title(u'花瓣长度和花瓣宽度')
    axs5_xlabel_text = axs[2][1].set_xlabel(u'花瓣长度')
    axs5_ylabel_text = axs[2][1].set_ylabel(u'花瓣宽度')
    plt.setp(axs5_title_text, size=9, weight='bold', color='red')
    plt.setp(axs5_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs5_ylabel_text, size=7, weight='bold', color='black')

    #设置图例
    setosa = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='setosa')   #设置setosa的图例为黑色的点，大小为6
    versicolor = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='versicolor')   #设置yersicolor的图例为橙色的点，大小为6
    virginica = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='virginica')  #设置virginica的图例为红色的点，大小为6

    axs[0][0].legend(handles=[setosa,versicolor,virginica])   #对每一个图形设置图例
    axs[0][1].legend(handles=[setosa,versicolor,virginica])
    axs[1][0].legend(handles=[setosa,versicolor,virginica])
    axs[1][1].legend(handles=[setosa, versicolor, virginica])
    axs[2][0].legend(handles=[setosa, versicolor, virginica])
    axs[2][1].legend(handles=[setosa, versicolor, virginica])
    #绘制图形
    plt.show()

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
    range = maxVal - minVal        #数据范围
    normAttributes = np.zeros(np.shape(attributes))  #初始化归一化数据
    m = attributes.shape[0]     #获取数据的行数
    normAttributes = attributes - np.tile(minVal, (m, 1))  #创建一个全是最小值得数组
    normAttributes = normAttributes / np.tile(range, (m, 1))  #创建一个全是范围值得数组
    return normAttributes   #返回归一化后的数据

def colcPer(attributes):
    """
    函数说明：计算每个特征之间的Person相关系数
    :parameter:
        attributes - 特征值
    :return: none
    """
    attributes = attributes.astype('float32')  #将attributes数据类型更改成float32
    colcPerAttributes = np.corrcoef(attributes, rowvar=0)   #以每一列计算相关系数
    colcPerAttributes = pd.DataFrame(data=colcPerAttributes, columns=['sl', 'sw', 'pl', 'pw'], index=['sl', 'sw', 'pl', 'pw'])  # 给数据集添加列名，方便观察
    print(colcPerAttributes)   #输出相关系数矩阵

if __name__ == '__main__':
    df = "iris.data"      #文件路径
    attributes, labels = loadDataSet(df)   #得到特征矩阵和标签变量
    showdatas(attributes, labels)      #输出相关的散点图
    normAttributes = autoNorm(attributes)  #进行归一化处理
    colcPer(normAttributes)    #计算相关系数