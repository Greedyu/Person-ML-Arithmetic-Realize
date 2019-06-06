# -*- coding: UTF-8 -*-
import random
import re

from os import listdir

def textParse(bigText):
    # re.split，支持正则及多个字符切割
    listOfWords = re.split(r'\W*',bigText)
    return [tok.lower() for tok in listOfWords if len(tok) > 2]

def createVocabList(dataSet):
    uniqueWord = set([])
    for wordList in dataSet:
        uniqueWord = set(wordList) | uniqueWord
    return list(uniqueWord)


def setOfWords2Vec(vocabList, wordList):
    vocabVec = [0] * len(vocabList)
    for word in wordList:
        if word in vocabList:
            vocabList[vocabVec.index(word)] = 1;
    return vocabVec;


if __name__ == '__main__':
    docList = []
    classList = []
    fullText = []
    fileDir = 'email/ham'
    fileDir1 = 'email/spam'
    trainingFileList = listdir(fileDir)
    m = len(trainingFileList)

    for i in range(m):
        wordList = textParse(open(fileDir + '/' +trainingFileList[i], 'r' , encoding= 'GBK').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)

    trainingFileList = listdir(fileDir1)
    m = len(trainingFileList)
    for i in range(m):
        wordList = textParse(open(fileDir1 + '/' + trainingFileList[i], 'r', encoding='GBK').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randomIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainingSet.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])