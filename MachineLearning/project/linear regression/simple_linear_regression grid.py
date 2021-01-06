# Simple Linear Regression 

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
parameters = {
    'fit_intercept':[True,False], 
    'normalize':[True,False], 
    'copy_X':[True, False]
    }

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
grid = GridSearchCV(regressor, parameters, n_jobs=-1, verbose=1)
grid.fit(X, y)
print(grid.best_params_)