import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.tree.tree import DecisionTreeRegressor

csvPath = 'data.csv'
preproType = 2
scoring = 'r2'
seed = 7


def getModels():
    models = {}
    models['dt'] = DecisionTreeRegressor(max_depth=50)
    models['rf1'] = RandomForestRegressor()
    models['rf2'] = RandomForestRegressor(n_estimators=128, max_depth=15)
    models['gbr'] = GradientBoostingRegressor(n_estimators=128, max_depth=5, learning_rate=1.0)
    # models['abr'] = AdaBoostRegressor(n_estimators=128)
    return models


def getXy():
    data = pd.read_csv(csvPath)
    labelIndex = len(data.columns) - 1
    print(labelIndex)
    print(data.columns.tolist())
    X = data.values[:, 0:labelIndex]
    y = data.values[:, labelIndex]

    return X, y


# 1: MinMaxScaler, 2:StandardScaler, 3:Normalizer
def preprocessingX(X, preproType):
    if preproType == 1:
        preproModel = MinMaxScaler()
    elif preproType == 2:
        preproModel = StandardScaler()
    elif preproType == 3:
        preproModel = Normalizer()
    else:
        raise Exception('Not supported proprocessing type:' + str(preproType))
    return preproModel.fit_transform(X)


def evaluation():
    X, y = getXy()
    newX = preprocessingX(X, preproType)
    models = getModels()

    for modelName in models:
        model = models[modelName]
        score = cross_val_score(model, newX, y, cv=KFold(n_splits=10, random_state=seed), scoring=scoring)
        print(modelName)
        print(score.mean())


def visualization():
    X, y = getXy()
    newX = preprocessingX(X, preproType)

    X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.33, random_state=seed)
    models = getModels()

    for modelName in models:
        model = models[modelName]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(len(y_predict))
        print(y_predict)
        x_axis = range(0, len(y_test))
        print(x_axis)
        plt.plot(x_axis, y_test, 'r', label='test')
        plt.plot(x_axis, y_predict, 'b', label='predict')
        #标签位置
        # plt.legend(bbox_to_anchor=[0.3, 1])
        plt.legend(loc=0)

        plt.grid()
        plt.show()


def realiseData():
    data = pd.read_csv(csvPath)
    # data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    pd.scatter_matrix(data)
    plt.show()


def test():
    X, y = getXy()
    # print(X[0:3, :])
    # print(y[0:3, :])
    # fit = SelectKBest(chi2, k=2).fit_transform(X, y)
    # print(fit.shape)
    rfe = RFE(RandomForestRegressor(), 3)
    fit = rfe.fit(X, y)
    print(fit.n_features_)
    print(fit.support_)
    print(fit.ranking_)


if __name__ == '__main__':
    evaluation()
    visualization()
    # realiseData()
    test()
