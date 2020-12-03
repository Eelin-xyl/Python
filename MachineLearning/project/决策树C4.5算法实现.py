import math
import numpy as np
import operator

'''
计算熵
输入:数据集
输出:熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]      				#获取训练集 标签
        if currentLabel not in labelCounts.keys():      #统计各 类别的次数
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries     #比值
        shannonEnt -= prob * math.log2(prob)        	#计算熵
    return shannonEnt

'''
用不同特征划分数据集
输入:
    dataSet    数据集
    axis       dataSet中的第几列
    value      划分的值(划分标准)
输出:以某个特征值划分得到的数据集(出去本次划分的特征列)
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            retDataSet.append(np.delete(featVec, axis))
    return np.array(retDataSet)
    

'''
求最大信息增益的特征列
输入:数据集
输出:信息增益最大特征列的索引
'''
def chooseBestFeatureTosplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       	#特征个数
    baseEntropy = calcShannonEnt(dataSet)       #总熵
    bestInfoIGR = 0.0
    bestFeature = -1
    for i in range(numFeatures):        		#循环的是 特征
        featList = [example[i] for example in dataSet]      
        uniqueVals = set(featList)              #计算各个特征中   类别个数
        newEntropy = 0.0
        for value in uniqueVals:        		#循环的是 特征中的 类别
            subDataSet = splitDataSet(dataSet, i, value)        #按特征划分数据集
            prob = len(subDataSet) / float(len(dataSet))        #计算 特征中 某个类型占所有的 比值
            newEntropy += prob * calcShannonEnt(subDataSet)     #在某个特征的划分下，新数据集的 总熵
        infoGain = baseEntropy - newEntropy             		#计算每种划分方式的熵(每个特征的信息增益)
#        计算每个特征的 信息熵
        infoMation = calcShannonEnt(dataSet[:, i].reshape(-1,1))
#         计算每个特征的信息增益率
        infoIGR = infoGain / infoMation
#         计算最大的信息增益率，并返回列索引
        if infoIGR > bestInfoIGR:
            bestInfoIGR = infoIGR
            bestFeature = i
    return bestFeature
	
'''
表决器
输入:类别标签组成的一个list
输出:某一个类别
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
    
'''
创建树
输入:
    dataSet:数据集
    labels:特征名称组成的list
输出:保存树信息的嵌套字典(因为是递归进行，所有是嵌套字典)
'''
def createTree(dataSet, labels):        					#labels特征名称
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #类别完全相同  停止划分
        return classList[0]
    if len(dataSet[0]) == 1:            					#遍历完所有特征
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureTosplit(dataSet)        	#计算信息增益最大的特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}             				#存储树的所有信息
    del(labels[bestFeat])           						#删除本次遍历   信息增益最大的特征名称
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)          				#得到本次遍历   信息增益最大特征的  所有值
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


'''
使用决策树 预测
输入:
    inputTree:保存树信息的字典
    featLabels:特征名称组成的list
    testVec:要预测的数据(单条、list)
输出:类别标签
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]            #取最外层 字典键
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)          #返回键在 标签中的索引位置
    for key in secondDict.keys():               	#循环  key
        if testVec[featIndex] == key:           	#此特征项   和  哪一个键 相等
            if type(secondDict[key]).__name__ == 'dict':        #查看是否为 叶节点
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


'''
使用决策树 预测，输入为 矩阵
输入:
    inputTree:保存树信息的字典
    featLabels:特征名称组成的list
    testMatrix:多条需要预测数据组成的ndarray
输出:多个类别标签组成的ndarray
'''
def classifyMa(inputTree, featLabels, testMatrix):
     
    testLabel = np.array([])
    for testVec in testMatrix:
        classLabel = classify(inputTree, featLabels, testVec)
        testLabel = np.append(testLabel, classLabel)
    return testLabel.reshape(-1,1)
	