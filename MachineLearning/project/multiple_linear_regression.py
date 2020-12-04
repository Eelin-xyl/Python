# Multiple Linear Regression

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

# Encoding categorical data
# Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()

# # Avoiding the Dummy Variable Trap
# X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
regressor = pl.make_pipeline(
    sp.PolynomialFeatures(4),
    Lasso()
)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print(mean_absolute_error(y_test, y_pred))

# Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
# X_train = np.append(arr = np.ones((40, 1)).astype(int), values = X_train, axis = 1)
# X_opt = X_train [:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = X_train [:, [0, 1, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = X_train [:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = X_train [:, [0, 3, 5]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()
# X_opt = X_train [:, [0, 3]]
# regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
# regressor_OLS.summary()








