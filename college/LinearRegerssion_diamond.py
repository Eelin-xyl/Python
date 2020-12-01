# -*- coding: utf-8 -*-

# 导入标准库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据
dataset = pd.read_csv('diamonds.csv')
# 查看导入数据的一些基本信息
dataset.info()
dataset.describe()

# 异常值处理，将x、y、z三列中为0的行删除
dataset = dataset[(dataset[['x','y','z']] != 0).all(axis=1)]

# 数据划分
X = dataset.iloc[:, [1,2,3,4,5,6,8,9,10]].values
y = dataset.iloc[:, 7].values

#没有缺失值

# 虚拟编码,同一特征值之间存在等级之分的 不用虚拟编码
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
X[:, 1] = labelEncoder.fit_transform(X[:, 1])
X[:, 2] = labelEncoder.fit_transform(X[:, 2])
X[:, 3] = labelEncoder.fit_transform(X[:, 3])
X = X.astype(np.float64)

# 数据划分   80%的训练集  20%的测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# 用训练集和选择的算法 拟合模型
from sklearn.linear_model import LinearRegression
regersson = LinearRegression()
regersson.fit(X_train, y_train)

# 用模型 测试测试集
y_pred = regersson.predict(X_test)

# 方向淘汰法选择特征
import statsmodels.api as sm
# 增加常数列
X_train = sm.add_constant(X_train)

# 计算所有特征的P值
X_opt = X_train
regersson_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regersson_OLS.summary()

# 将P值最大的特征删除，继续计算剩下特征的P值
X_opt = X_train[:, [0,1,2,3,4,5,6,7,8]]
regersson_OLS = sm.OLS(endog= y_train, exog = X_opt).fit()
regersson_OLS.summary()

