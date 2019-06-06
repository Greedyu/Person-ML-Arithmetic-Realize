from _csv import reader

import numpy
from matplotlib import pyplot
from pandas import read_csv, set_option
from sklearn.preprocessing import MinMaxScaler, StandardScaler

filename = '/Users/dongsheng/Documents/me/resource/MachineLearning-master/chapter05/pima_data.csv'
with open (filename , 'rt') as raw_data:
    readers = reader(raw_data,delimiter=',')
    x = list(readers)
    data = numpy.array(x).astype('float')

    print(data.shape)

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

array = data.values
# X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据
X = array[:,0:8]
Y = array[:,8]
# print(X)
# 调整数据尺度
# transformer = MinMaxScaler(feature_range=(0,1))
# newX = transformer.fit_transform(X)
# numpy.set_printoptions(precision=3)
# 正态化数据Standardize Data
transformer = StandardScaler().fit(X)
newX = transformer.transform(X)
print(newX)

print(len(newX))
# print(data.head(10))

# set_option('display.width',100)
#精确到小数点几位
# set_option('precision',3)
# 数据的维度
# print(data.groupby('class').size())
#描述性统计
# print(data.describe())
#计算数据相关性
# print(data.corr(method='pearson'))
# 数据的分布分析
# print(data.skew())


#直方图
# data.hist()
# pyplot.show();
#密度图 sharex 自适应，subplots ：每个特征一个面板
# data.plot(kind='density',layout=(3,3),subplots=True,sharex=False)
#箱线图
# data.plot(kind='density',layout=(3,3),subplots=True,sharex=False)
# pyplot.show();