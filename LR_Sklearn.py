#!/usr/bin/env python
#-*-coding:utf-8 -*-
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c


def loadData(filename):
    fr = open(filename)
    dataSet = [];dataLables = []
    for currLine in fr.readlines():
        line = currLine.strip().split('\t')
        lineArr = []
        for i in range(len(line) - 1):
            lineArr.append(float(line[i]))
        lineArr.append(1.0) #对应常数b的那一列
        dataSet.append(lineArr)
        dataLables.append(float(line[-1]))
    return dataSet, dataLables


if __name__ == '__main__':
    trainDataSet, trainLabels = loadData('horseColicTraining.txt')
    testDataSet, testLabels = loadData('horseColicTest.txt')
    # LogisticRegression同样实现了fit()和predict()方法
    classifier =linear_model. LogisticRegression()
    classifier.fit(trainDataSet, trainLabels)
    predicLabels = classifier.predict(testDataSet)
    m = len(testLabels)

    errorCount = 0
    for i in range(m):
        if predicLabels[i] != testLabels[i]:
            errorCount += 1
    errorRate = float(errorCount) / float(m)
    print 'm %d' % m
    print 'errorCount %d' % errorCount
    print 'errorRate %f' % errorRate

