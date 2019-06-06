from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

filename = '/Users/dongsheng/Documents/me/resource/MachineLearning-master/chapter05/pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

array = data.values
# X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据
X = array[:,0:8]
Y = array[:,8]
size = 0.33
seed = 4

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = size,random_state=seed)
# 逻辑回归是分类模型（不是回归模型）。
model = LogisticRegression()
# model.fit(X_train,Y_train)
# result = model.score(X_test,Y_test)
# print("算法评估结果：%.3f%%" % (result * 100))

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)
result = cross_val_score(model,X,Y,cv=kfold)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100 , result.std() * 100))
