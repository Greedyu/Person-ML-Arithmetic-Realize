def calcShannonEnt(dataSet):
    # '''
    # 计算香农熵
    # :param dataSet:数据集
    # :return: 计算结果
    # '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # 遍历每个实例，统计标签的频数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        from math import log
        shannonEnt -= prob * log(prob,2) # 以2为底的对数
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    # '''
    # 按照给定特征划分数据集
    # :param dataSet:待划分的数据集
    # :param axis:划分数据集的特征
    # :param value: 需要返回的特征的值
    # :return: 划分结果列表
    # '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    # '''
    # 计算X_i给定的条件下，Y的条件熵
    # :param dataSet:数据集
    # :param i:维度i
    # :param featList: 数据集特征列表
    # :param uniqueVals: 数据集特征集合
    # :return: 条件熵
    # '''
    conditionEnt = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        conditionEnt += prob * calcShannonEnt(subDataSet)  # 条件熵的计算
    return conditionEnt

def calcInformationGain(dataSet, baseEntropy, i):
    # '''
    # 计算信息增益
    # :param dataSet:数据集
    # :param baseEntropy:数据集的信息熵
    # :param i: 特征维度i
    # :return: 特征i对数据集的信息增益g(D|X_i)
    # '''
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就yes熵的减少，也就yes不确定性的减少
    return infoGain

def calcInformationGainRatio(dataSet, baseEntropy, i):
    # '''
    # 计算信息增益比
    # :param dataSet:数据集
    # :param baseEntropy:数据集的信息熵
    # :param i: 特征维度i
    # :return: 特征i对数据集的信息增益比gR(D|X_i)
    # '''
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy