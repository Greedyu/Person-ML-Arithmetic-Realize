# -*- coding: UTF-8 -*-
import numpy as np


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

def reduceRepeatString(postringList):
    vocabSet = set([])
    for document in postringList:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def SetOfWords2Vec(postringList , postinDoc):
    wordsVec = [0] * len(postringList)
    for word in postinDoc:
        if word in postringList:
            wordsVec[postringList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return wordsVec


def trainNB0(trainMat, classVec):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pAbusive = sum(classVec) / float(trainMat)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom= 2.0
    p1Denom= 2.0
    for i in range(numTrainDocs):
        if classVec[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Num += sum(trainMat[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive




if __name__ == '__main__':
    postringList ,classVec = loadDataSet()

    uniqueList = reduceRepeatString(postringList)
    trainMat = []
    for postinDoc in postringList:
        trainMat.append(SetOfWords2Vec(postringList,postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)