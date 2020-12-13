# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:38:18 2018

@author: aoanng
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(finalcleaned).csv', encoding='gbk')

dataset = dataset.iloc[:,:].values

X = dataset[:, 1:]
y = dataset[:, 0]

##创建100个类共10000个样本，每个样本10个特征
X, y = make_blobs(n_samples=len(X), n_features=12, centers=26,random_state=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(len(X_train))

## 决策树
clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores1 = cross_val_score(clf1, X, y)
print(scores1.mean())

## 随机森林
clf2 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores2 = cross_val_score(clf2, X, y)
print(scores2.mean())

## ExtraTree分类器集合
clf3 = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores3 = cross_val_score(clf3, X, y)
print(scores3.mean())