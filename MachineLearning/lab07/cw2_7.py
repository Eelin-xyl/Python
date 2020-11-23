import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error


# CW2.7
def mse_of_linear_svr(X, y, epsilon):
    """
        Compute the mean square error of a linear SVR predictor with hyperparameter epsilon.
        As a model, use LinearSVR library to train a linear SVR predictor.
        Set its epsilon hyperparameter to the value of the epsilon argument,
            and its random state to 5.

        Split the dataset into training dataset, test dataset, training labels, and test labels;
            with 0.2 as the test size and 5 as its random state.
        Use StandardScaler to scale the both datasets.

        Fit and test the model, and return the mean square error on the test dataset.

        Args:
            X - (n, d) numpy array of the dataset of n sample points each with d features
            y - (n, ) numpy array of the label values for each sample point
            epsilon - a scalar of the hyperparameter epsilon of a linear SVR predictor
        Returns:
            mse - a scalar of the mean square error of the test dataset
    """
    # Write your code here
    model_linearSVR = LinearSVR(epsilon = epsilon,random_state=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    scaler = StandardScaler()
    scaler_x_train = scaler.fit_transform(X_train)
    scaler_x_test = scaler.transform(X_test)
    model_linearSVR.fit(scaler_x_train, y_train)
    y_pred = model_linearSVR.predict(scaler_x_test)
    a = mean_squared_error(y_test, y_pred)
    return a


def linear_svr_on_boston_housing_dataset():
    X, y = datasets.load_boston(return_X_y=True)
    epsilon = 1.5
    print(mse_of_linear_svr(X, y, epsilon))


def main():
    linear_svr_on_boston_housing_dataset()


if __name__ == "__main__":
    main()