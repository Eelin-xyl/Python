import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\MachineLearning\\project\\SpotifyAudioFeaturesApril2019.csv')

dataset=dataset.drop('artist_name',axis=1)
dataset=dataset.drop('track_name',axis=1)
dataset=dataset.drop('track_id',axis=1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        'weights' : ['uniform'],
        'n_neighbors' : [i for i in range(1, 100, 20)]
    },
    {
        'weights' : ['distance'],
        'n_neighbors' : [j for j in range(1, 100, 20)],
        'p' : [n for n in range(1, 20, 2)]
    }
]

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor()
grid_search = GridSearchCV(knn_model, param_grid, n_jobs = -1, verbose = 1)
grid_search.fit(X, y)
print(grid_search.best_params_)



# from sklearn.metrics import mean_absolute_error
# y_pred = knn_model.predict(X_test)
# print(mean_absolute_error(y_test, y_pred))