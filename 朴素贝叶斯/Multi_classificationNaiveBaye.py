import numpy as np
from collections import Counter
from sklearn import datasets
import pickle
import time
from scipy import sparse

class NaiveBayes:
    def __init__(self, lamb=1):
        self.lamb = lamb  # 贝叶斯估计的参数
        self.prior = dict()  # 存储先验概率
        self.conditional = dict()  # 存储条件概率


    def training(self, features, target):
        """
        根据朴素贝叶斯算法原理,使用 贝叶斯估计 计算先验概率和条件概率
        特征集集为离散型数据,预测类别为多元.  数据集格式为np.array
        :param features: 特征集m*n,m为样本数,n为特征数
        :param target: 标签集m*1
        :return: 不返回任何值,更新成员变量
        """
        features = np.array(features)
        target = np.array(target).reshape(features.shape[0], 1)
        m, n = features.shape    #得到样本数和特征树
        labels = Counter(target.flatten().tolist())  # 计算各类别的样本个数
        k = len(labels.keys())  # 类别数
        for label, amount in labels.items():
            #进行了拉普拉斯修正
            self.prior[label] = (amount + self.lamb) / (m + k * self.lamb)  # 计算平滑处理后的先验概率
        for feature in range(n):  # 遍历每个特征
            self.conditional[feature] = {}
            values = np.unique(features[:, feature])
            for value in values:  # 遍历每个特征值
                self.conditional[feature][value] = {}    #每个特征的值
                for label, amount in labels.items():  # 遍历每种类别
                    feature_label = features[target[:, 0] == label, :]  # 截取该类别的数据集
                    c_label = Counter(feature_label[:, feature].flatten().tolist())  # 计算该类别下各特征值出现的次数
                    self.conditional[feature][value][label] = (c_label.get(value, 0) + self.lamb) / (amount + len(values) * self.lamb)  # 计算平滑处理后的条件概率
        return

    def predict(self, features):
        """预测单个样本"""
        best_poster, best_label = -np.inf, -1
        for label in self.prior:
            poster = np.log(self.prior[label])  # 初始化后验概率为先验概率,同时把连乘换成取对数相加，防止下溢（即太多小于1的数相乘，结果会变成0）
            for feature in range(features.shape[0]):
                poster += np.log(self.conditional[feature][features[feature]][label])
            if poster > best_poster:  # 获取后验概率最大的类别
                best_poster = poster
                best_label = label
        return best_label


def test():
    dataset = datasets.load_iris()  # 鸢尾花数据集
    dataset = np.concatenate((dataset['data'], dataset['target'].reshape(-1, 1)), axis=1)  # 组合数据
    np.random.shuffle(dataset)  # 打乱数据
    features = dataset[:, :-1]
    target = dataset[:, -1:]
    train_features = features[:100]
    train_target = target[:100]
    test_features = features[-20:]
    test_target = target[-20:]
    nb = NaiveBayes()
    nb.training(train_features, train_target)
    prediction = []
    for features in test_features:
        prediction.append(nb.predict(features))
    correct = [1 if a == b else 0 for a, b in zip(prediction, test_target)]
    print(correct.count(1) / len(correct))  # 计算准确率


if __name__ == '__main__':
    fr = open('Data/4news_100.pkl', 'rb')
    inf = pickle.load(fr)
    features = inf.toarray()

    fr1 = open('Data/label_names.pkl', 'rb')
    inf1 = pickle.load(fr1)

    fr2 = open('Data/labels.pkl', 'rb')
    target = pickle.load(fr2)
    dataset = np.concatenate((features, target.reshape(-1, 1)), axis=1)  # 组合数据
    #np.random.shuffle(dataset)  # 打乱数据
    features = dataset[:, :-1]
    target = dataset[:, -1:]
    train_features = features[:100]
    train_target = target[:100]
    test_features = features[-100:]
    test_target = target[-100:]
    time1 = time.clock()
    nb = NaiveBayes()
    nb.training(train_features, train_target)
    time2 = time.clock()
    print(time2-time1)
    prediction = []
    for features in test_features:
        prediction.append(nb.predict(features))
    correct = [1 if a == b else 0 for a, b in zip(prediction, test_target)]
    print(correct.count(1) / len(correct))  # 计算准确率