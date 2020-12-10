# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures3.csv',encoding='gbk')

# dataset=dataset.drop('artist_name',axis=1)
dataset=dataset.drop('track_name',axis=1)
# dataset=dataset.drop('key',axis=1)
dataset=dataset.drop('track_id',axis=1)

dataset = dataset.iloc[:50000,:].values

X = dataset[:, 1:]
y = dataset[:, 0]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder = LabelEncoder()
X[:,7] = labelEncoder.fit_transform(X[:,7])

ct = ColumnTransformer([('artist_name', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 125, criterion = 'gini', random_state = 0 )

# classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=50, min_samples_split=10, 
# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
# min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
# random_state=None, verbose=0, warm_start=False, class_weight=None)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test)*100, '%')