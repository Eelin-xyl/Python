import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Python\\MachineLearning\\project\\SpotifyAudioFeaturesApril2019.csv')

# dataset=dataset.drop('artist_name',axis=1)
dataset=dataset.drop('track_name',axis=1)
dataset=dataset.drop('track_id',axis=1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('artist_name', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg = KNeighborsRegressor(n_neighbors = 10, p = 1, weights = 'distance')
knn_reg.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error
y_pred = knn_reg.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

#Visualization
fig = plt.figure(figsize=(10,6))
plt.plot(range(y_test.shape[0]),y_test,color="blue", linewidth=1.5, linestyle="-")
plt.plot(range(y_test.shape[0]),y_pred,color="red", linewidth=1.5, linestyle="-.")
plt.legend(['True','Pre'])
plt.show()