该数据取自20newsgroup数据集的4个类别的共100条数据
4news_100.pkl: 进行了词袋统计处理，每一行为一则新闻，每一列代表各个关键词的出现次数，numpy稀疏矩阵，基本可当numpy array使用
labels.pkl: 目标类别，共四类，list
label_names.pkl: 目标类别对应的名称，dict
以上文件使用pickle库导入