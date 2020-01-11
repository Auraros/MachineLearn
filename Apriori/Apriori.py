# -*- coding:utf-8 -*-
import warnings
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')   #忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def loadDataSet():
    #读入文件
    dataSet = pd.read_csv('agaricus-lepiota.data', ',', header=None)
    #加上列名，便于操作
    dataSet = pd.DataFrame(data=np.array(dataSet),columns=['classes', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape','stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'], index=range(8124))

    model_label01 = dataSet["classes"].replace({"e": '1', "p":'2' })
    model_label02 = dataSet["cap-shape"].replace({"b":'3', "c":'4', "x":'5', "f":'6', "k":'7', "s":'8'})
    model_label03 = dataSet["cap-surface"].replace({"f":'9', "g":'10', "y":'11', "s":'12'})
    model_label04 =  dataSet["cap-color"].replace({"n":'13', "b":'14', "c":'15', "g":'16', "r":'17', "p":'18', "u":'19', "e":'20', "w":'21', "y":'22'})
    model_label05 =  dataSet["bruises"].replace({"t":'23', "f":'24'})
    model_label06 =  dataSet["odor"].replace({"a":'25', "l":'26', "c":'27', "y":'28', "f":'29', "m":'30', "n":'31', "p":'32', "s":'33'})
    model_label07 =  dataSet['gill-attachment'].replace({"a":'34', "d":'35', "f":'36', "n":'37'})
    model_label08 =  dataSet["gill-spacing"].replace({"c":'38', "w":'39', "d":'40'})
    model_label09 =  dataSet["gill-size"].replace({"b":'41', "n":'42'})
    model_label10 =  dataSet["gill-color"].replace({"k":'43', "n":'44', "b":'45', "h":'46', "g":'47', "r":'48', "o":'49', "p":'50', "u":'51', "e":'52', "w":'53', "y":'54'})
    model_label11 =  dataSet['stalk-shape'].replace({"e":'55', "t":'56'})
    model_label12 =  dataSet["stalk-root"].replace({"b":'57', "c":'58', "u":'59', "e":'60', "z":'61', "r":'62', "?":'63'})
    model_label13 =  dataSet["stalk-surface-above-ring"].replace({"f":'64', "y":'65', "k":'66', "s":'67'})
    model_label14 =  dataSet["stalk-surface-below-ring"].replace({"f": '68', "y": '69', "k": '70', "s": '71'})
    model_label15 =  dataSet["stalk-color-above-ring"].replace({"n":'72', "b":'73', "c":'74', "g":'75', "o":'76', "p":'77', "e":'78', "w":'79', "y":'80'})
    model_label16 =  dataSet["stalk-color-above-ring"].replace({"n": '81', "b": '82', "c": '83', "g": '84', "o": '85', "p": '86', "e": '87', "w": '88', "y": '89'})
    model_label17 =  dataSet["veil-type"].replace({"p":'90', "u":'91'})
    model_label18 =  dataSet["veil-color"].replace({"n":'92', "o":'93', "w":'94', "y":'95'})
    model_label19 =  dataSet["ring-number"].replace({"n":'96', "o":'97', "t":'98'})
    model_label20 =  dataSet["ring-type"].replace({"c":'99', "e":'100', "f":'101', "l":'102', "n":'103', "p":'104', "s":'105', "z":'106'})
    model_label21 =  dataSet["spore-print-color"].replace({"k":'107', "n":'108', "b":'109', "h":'110', "r":'101', "o":'102', "u":'103', "w":'104', "y":'105'})
    model_label22 =  dataSet["population"].replace({"a":'106', "c":'107', "n":'108', "s":'109', "v":'110', "y":'111'})
    model_label23 =  dataSet["habitat"].replace({"g":'112', "l":'113', "m":'114', "p":'115', "u":'116', "w":'117', "d":'118'})
    model_label = pd.concat([model_label01, model_label02, model_label03, model_label04, model_label05, model_label06, model_label07, model_label08, model_label09, model_label10, model_label11, model_label12, model_label13, model_label14, model_label15, model_label16, model_label17, model_label18, model_label19, model_label20, model_label21, model_label22, model_label23],axis = 1,  ignore_index=False)
    return model_label


def createC1(dataSet):
    """
    构建初始候选项集的列表，即所有候选项集只包含一个元素，
    C1是大小为1的所有候选项集的集合
    :param dataSet:数据集
    :return:
    """
    #定义候选集列表C1
    C1 = []
    #遍历数据集合，并且遍历每一个集合中的每一项，创建只包含一个元素的候选项集集合
    for transaction in dataSet:
        for item in transaction:
            # 如果没有在C1列表中，则将该项的列表形式添加进去
            if not [item] in C1:
                C1.append([item])
    # 对列表进行排序
    C1.sort()
    # 固定列表C1，使其不可变
    return list(map(frozenset, C1))


def scanD(D,Ck,minSupport):
    """
    函数说明:创建满足支持度要求的候选键集合
    """
    # 定义存储每个项集在消费记录中出现的次数的字典
    ssCnt={}
    # 遍历这个数据集，并且遍历候选项集集合，判断候选项是否是一条记录的子集
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                # 如果是则累加其出现的次数
                if not can in ssCnt:
                    ssCnt[can]=1
                else: ssCnt[can]+=1
    # 计算数据集总及记录数
    numItems=float(len(D))
    # 定义满足最小支持度的候选项集列表
    retList = []
    # 用于所有项集的支持度
    supportData = {}
    # 遍历整个字典
    for key in ssCnt:
        # 计算当前项集的支持度
        support = ssCnt[key]/numItems
        # 如果该项集支持度大于最小要求，则将其头插至L1列表中
        if support >= minSupport:
            retList.insert(0,key)
        # 记录每个项集的支持度
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    函数说明：#上述函数创建了L1，则现在需要创建由L1->C2的函数，也就是说需要将每个项集集合元素加1
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    """
    # 存储Ck的列表
    retList = []
    # 获取lkPri长度，便于在其中遍历
    lenLk = len(Lk)
    # 两两遍历候选项集中的集合
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 因为列表元素为集合，所以在比较前需要先将其转换为list,选择集合中前k-2个元素进行比较，如果相等，则对两个集合进行并操作
            # 这里可以保证减少遍历次数，并且可保证集合元素比合并前增加一个
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            # 对转化后的列表进行排序，便于比较
            L1.sort(); L2.sort()
            if L1==L2: #若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


def apriori(dataSet, minSupport = 0.5):
    """
    函数说明：生成所有频繁项集函数
    """
    # 创建C1
    C1 = createC1(dataSet)
    # 对数据集进行转换，并调用函数筛选出满足条件的项集
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)#单项最小支持度判断 0.5，生成L1
    # 定义存储所有频繁项集的列表
    L = [L1]
    k = 2
    # 迭代开始，生成所有满足条件的频繁项集（每次迭代项集元素个数加1）
    # 迭代停止条件为，当频繁项集中包含了所有单个项集元素后停止
    while (len(L[k-2]) > 0):#创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck = aprioriGen(L[k-2], k)#Ck
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        # 更新supportData
        # 不断的添加以项集为key，以项集的支持度为value的元素
        # 将此次迭代产生的频繁集集合加入L中
        L.append(Lk)
        k += 1
    return L, supportData

#生成关联规则
def generateRules(L, supportData, minConf=0.7):
    #频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = [] #存储所有的关联规则
    for i in range(1, len(L)):  #只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
            #如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)# 调用函数2
    return bigRuleList

#生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #针对项集中只有两个元素时，计算可信度
    prunedH = []#返回一个满足最小可信度要求的规则列表
    for conseq in H:#后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #可信度计算，结合支持度数据
        if conf >= minConf:
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            #如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet-conseq, conseq, conf))#添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)#同样需要放入列表到后面检查
    return prunedH

#合并
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m+1)#存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        if (len(Hmp1) > 1):
        #满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

if __name__ == '__main__':
    dataSet = loadDataSet()
    dataSet = dataSet.values
    dataSet = dataSet.tolist()
    t1 = time()
    L, suppData = apriori(dataSet, minSupport=0.7)
    rules = generateRules(L, suppData, minConf=0.7)
    t2 = time()
    time = t2 - t1
    print(f"耗时：{time}秒")
