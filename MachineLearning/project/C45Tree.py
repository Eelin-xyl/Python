'''
Created on 2019年5月22日

@author: FG

Tree C4.5
'''

import math
import pandas as pd
import numpy as np
import operator


'''
计算信息熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]      #获取训练集 标签
        if currentLabel not in labelCounts.keys():      #统计各 类别的次数
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries     #比值
        shannonEnt -= prob * math.log2(prob)        #计算熵
    return shannonEnt

'''
用不同特征划分数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            retDataSet.append(np.delete(featVec, axis))     #出去本特征
#    将list转为ndarray
    return np.array(retDataSet)
    

'''
计算各个特征的信息增益
'''
def chooseBestFeatureTosplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       #特征个数
    baseEntropy = calcShannonEnt(dataSet)       #总熵
    bestInfoIGR = 0.0
    bestFeature = -1
    for i in range(numFeatures):        #循环的是 特征
        featList = [example[i] for example in dataSet]      
        uniqueVals = set(featList)              #计算各个特征中   类别个数
        newEntropy = 0.0
        for value in uniqueVals:        #循环的是 特征中的 类别
            subDataSet = splitDataSet(dataSet, i, value)        #按特征划分数据集
            prob = len(subDataSet) / float(len(dataSet))        #计算 特征中 某个类型占所有的 比值
            newEntropy += prob * calcShannonEnt(subDataSet)     #在某个特征的划分下，新数据集的 总熵
        infoGain = baseEntropy - newEntropy             #计算每种划分方式的熵(每个特征的信息增益)
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
'''
def createTree(dataSet, labels):        #labels特征名称
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #类别完全相同  停止划分
        return classList[0]
    if len(dataSet[0]) == 1:            #遍历完所有特征
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureTosplit(dataSet)        #计算信息增益最大的特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}             #存储树的所有信息
    del(labels[bestFeat])           #删除本次遍历   信息增益最大的特征名称
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)          #得到本次遍历   信息增益最大特征的  所有值
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


'''
使用决策树 预测
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]            #取最外层 字典键
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)          #返回键在 标签中的索引位置
    for key in secondDict.keys():               #循环  key
        if testVec[featIndex] == key:           #此特征项   和  哪一个键 相等
            if type(secondDict[key]).__name__ == 'dict':        #查看是否为 叶节点
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
#         else:
#             classLabel = '无'
    return classLabel


'''
使用决策树 预测，输入为 矩阵
'''
def classifyMa(inputTree, featLabels, testMatrix):
     
    testLabel = np.array([])
    for testVec in testMatrix:
        classLabel = classify(inputTree, featLabels, testVec)
        testLabel = np.append(testLabel, classLabel)
    return testLabel.reshape(-1,1)
    



'''
存在漏洞，如果测试集中出现了训练集中不存在的特征，会报错
'''

# data = pd.read_csv(r'C:\Users\FG\Desktop\curriculum\tree_test.csv')
# data.drop(columns=['ID'], inplace=True)
# dataSet = data.values
# 
# print(chooseBestFeatureTosplit(dataSet))

from sklearn.model_selection import train_test_split
 
data = pd.read_csv(r'C:\Users\FG\Desktop\curriculum\tree_test.csv')
data.drop(columns=['ID'], inplace=True)
labels1 = list(data.columns)
labels2 = list(data.columns)
dataSet = data.values

#train, test = train_test_split(dataSet, test_size=0.2, random_state=1)
#print(train)
#print(test)
 
myTree = createTree(dataSet, labels1)
print(myTree)
# pred = classifyMa(myTree, labels2, test)
# print(pred)


# 蘑菇分类
#df = pd.read_csv(r'tree_test.csv')
#
#z = df[df['有自己的房子'] == '是']
#v = df[df['有自己的房子'] == '否']
#z = z.iloc[:, [0,1,2,4,5]]
#v = v.iloc[:, [0,1,2,4,5]]
#
#v1 = v[v['有工作'] == '是']
#v2 = v[v['有工作'] == '否']
#v1 = v1.iloc[:, [0,1,3,4]]
#v2 = v2.iloc[:, [0,1,3,4]]
#
#data = df.iloc[:,1:]
#data['class'] = df.iloc[:, 0]


# X = data.iloc[:, 0:-1].values
# y = data.iloc[:, -1].values
#  
# from sklearn.preprocessing import LabelEncoder
# labelEncoder = LabelEncoder()
# for i in range(20):
#     X[:, i] = labelEncoder.fit_transform(X[:, i])
# 
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)



# labels1 = list(data.columns)
# labels2 = list(data.columns)
# data = data.values
# train, test = train_test_split(data, test_size=0.2, random_state=0) 
# myTree = createTree(train, labels1)
# # print(myTree)
# pred = classifyMa(myTree, labels2, test)
# print(pred[:10])
# print(test[:, -1][:10])
#  
#  
#  
# from sklearn.metrics import confusion_matrix
# # cm = confusion_matrix(test[:, -1], pred)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)


