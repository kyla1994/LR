#-*-coding:utf-8 -*-
import math
import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

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

class LogisRegres(object):
    def __init__(self, fileTrain, fileTest):
        self.fileTrain = fileTrain
        self.fileTest = fileTest
        self.errorRate = 0


    def trainGA(self, dataSet, labels):
        dataMat = np.mat(dataSet)
        labelMat = np.mat(labels).transpose()
        m, n = dataMat.shape # m:样例数，n：特征数
        weights = np.ones((n, 1))
        alpha = 0.001
        maxcycles = 500

        for k in range(maxcycles):
            h = sigmoid(dataMat * weights)
            error = labelMat - h
            weights = weights + alpha * dataMat.transpose() * error #GD梯度上升法
        return weights

    def trainSGA0(self, dataSet, labels):
        dataMat = np.mat(dataSet)
        labelMat = np.mat(labels).transpose()
        m, n = dataMat.shape # m:样例数，n：特征数
        weights = np.ones((n, 1))
        alpha = 0.01

        for i in range(m):
            h = sigmoid(sum(dataMat[i] * weights))
            error = labelMat[i] - h
            weights = weights + alpha * dataMat[i].transpose() * error #GD梯度上升法
        return weights

    def trainSGA1(self, dataSet, labels, maxIter = 150):
        dataMat = np.mat(dataSet)
        labelMat = np.mat(labels).transpose()
        m, n = dataMat.shape # m:样例数，n：特征数
        weights = np.ones((n, 1))


        for k in range(maxIter):
            dataIndex = range(m)
            for i in range(m):
                alpha = 4.0 / (1.0 + i + k) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                h = sigmoid(sum(dataMat[randIndex] * weights))
                error = labelMat[randIndex] - h
                weights = weights + alpha * dataMat[randIndex].transpose() * error #GD梯度上升法
                del dataIndex[randIndex]
        return weights

    def classifyVec(self, dataSet, weights):
        dataMat = np.mat(dataSet)
        m = dataMat.shape[0]
        prob = sigmoid(dataMat * weights)
        labels = []
        for i in range(m):
            if prob[i] < 0.5:
                labels.append(float(0))
            else:
                labels.append(float(1))
        return labels


    def colicTrain(self):
        trainSet, trainLables = loadData(self.fileTrain)
        #weightsGA = self.trainGA(trainSet, trainLables)
        #weightsSGA0 = self.trainSGA0(trainSet, trainLables)
        weightsSGA1 = self.trainSGA1(trainSet, trainLables)
        return weightsSGA1

    def colicTest(self, weights):
        testSet, testLabels = loadData(self.fileTest)
        errorCount = 0
        labels = self.classifyVec(testSet, weights)
        for i in range(len(testLabels)):
            if labels[i] != testLabels[i]:
                errorCount += 1
        print errorCount
        print len(testLabels)
        self.errorRate = float(errorCount) / len(testLabels)
        print 'errorRate % f' % self.errorRate


if __name__ == '__main__':
    LR = LogisRegres('horseColicTraining.txt', 'horseColicTest.txt')
    weights = LR.colicTrain()
    errorRate = LR.colicTest(weights)





