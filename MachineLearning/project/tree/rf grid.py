# Random Forest Classification

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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(criterion = 'gini', n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
# Build a classification task using 3 informative features

# rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
# bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)



param_grid = { 
    'n_estimators': [10, 50, 100, 150, 200, 250],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)