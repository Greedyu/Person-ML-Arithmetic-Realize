import operator
import pickle

from math import log


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels =  ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
    return dataSet,labels

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    # key 获取排序的键值
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabal = featVec[-1]
        if currentLabal not in labelCounts.keys():
            labelCounts[currentLabal] = 0
        labelCounts[currentLabal] += 1
    shannonEnt = 0
    for label in labelCounts:
        prob = int(labelCounts[label]) / numEntires
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    subDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            subDataSet.append(reducedFeatVec)
    return subDataSet

def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = -1
    bestFeature = 0
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        featSet = set(featList)
        cdtEntropy = 0
        for value in featSet:
            subDataSet = splitDataSet(dataSet,i,value)
            prod = len(subDataSet) / float(len(dataSet))
            cdtEntropy += prod * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - cdtEntropy
        if (infoGain > bestInfoGain) :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature;


def createTree(dataSet ,labels,featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet[0])
    bestFeature = chooseBestFeature(dataSet)
    bestFeatLabel = lables[bestFeature]
    featLabels.append(bestFeatLabel)
    mytree = {bestFeatLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    featSet = set(featValues)
    for value in featSet:
        mytree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),labels,featLabels)
    return mytree

def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)
    classLabel = 'no'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    dataSet,lables = createDataSet()
    featLabels = []
    myTree = createTree(dataSet,lables,featLabels)
    storeTree(myTree,"myTree.txt")
    tree = grabTree("myTree.txt")
    print(tree)
    testVec = [0, 1]  # 测试数据
    result = classify(tree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')