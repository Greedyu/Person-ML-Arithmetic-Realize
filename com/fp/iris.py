from matplotlib import pyplot
from pandas import read_csv, scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

filename = "/Users/dongsheng/Documents/me/resource/MachineLearning-master/chapter03/iris.data.csv"
names = ['separ-length','separ-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

print('数据维度: 行 %s，列 %s' % dataset.shape)
# print(dataset.head(10))
# print(dataset.describe())
print(dataset.groupby('class').size())

# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# dataset.hist()
# scatter_matrix(dataset)
# pyplot.show()

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=validation_size, random_state=seed)

print(X_validation)
print(Y_validation)
# 算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    # results.append(cv_results)
    print('%s: %f (%f)' %(key, cv_results.mean(), cv_results.std()))

