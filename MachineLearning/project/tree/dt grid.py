# Decision Tree Classification

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

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
pipe_tree = make_pipeline(DecisionTreeRegressor())

# make an array of depths to choose from, say 1 to 20
depths = np.arange(1, 21)
num_leafs = [1, 5, 10, 20, 50, 100]

from sklearn.model_selection import GridSearchCV
param_grid = [{'decisiontreeregressor__max_depth':depths,
              'decisiontreeregressor__min_samples_leaf':num_leafs}]
gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, verbose=1, n_jobs=-1)
gs = gs.fit(X, y)
print(gs.best_params_)