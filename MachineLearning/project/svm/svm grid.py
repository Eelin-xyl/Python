# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\MachineLearning\\project\\SpotifyAudioFeaturesApril2019.csv')

dataset=dataset.drop('artist_name',axis=1)
dataset=dataset.drop('track_name',axis=1)
dataset=dataset.drop('track_id',axis=1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import GridSearchCV
param_grid = {
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'C' : [1, 5, 10],
    'degree' : [3, 8],
    'coef0' : [0.01, 10, 0.5],
    'gamma' : ['auto', 'scale']
    }

# Fitting regression to the Training set
from sklearn import svm
model_SVR = svm.SVR()
grids = GridSearchCV(model_SVR, param_grid, n_jobs = -1, verbose = 1)
grids.fit(X, y)
print(grids.best_params_)

# Predicting the Test set results
# from sklearn.metrics import mean_absolute_error
# y_pred = model_SVR.predict(X_test)
# print(mean_absolute_error(y_test, y_pred))