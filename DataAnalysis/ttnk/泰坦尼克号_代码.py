import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
data_train = pd.read_csv(r'Python\DataAnalysis\ttnk\train.csv')

pd.set_option('display.max_columns', None)
data_train.describe()
data_train.info()

# 数据划分，并对特征进行初步筛选
X = data_train.iloc[:,[2,4,5,6,7,11]].values
y = data_train.iloc[:,1].values

# 缺失值处理
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X[:,2:3])
X[:,2:3] = imp_mean.transform(X[:,2:3])

imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean.fit(X[:,5:6])
X[:,5:6] = imp_mean.transform(X[:,5:6])

# 虚拟编码
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,1] = labelEncoder.fit_transform(X[:,1])
X[:,5] = labelEncoder.fit_transform(X[:,5])

oneHotEncoder = OneHotEncoder(categorical_features=[5])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]

# 划分数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 建模
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 测试
y_pred = classifier.predict(X_test)

# 计算正确值
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
