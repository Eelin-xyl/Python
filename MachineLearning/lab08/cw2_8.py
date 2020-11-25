import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


# CW2 8
def pred_prob_softmax_reg(X_train, y_train, C, X_test):
    """
        Compute the predictions and the classes prediction probabilities of a test set
            classified using softmax regression model.

        With LogisticRegression function,
            set its the multi_class hyperparameter to "multinomial"
            set the solver as "lbfgs"
            set the regularization hyperparameter C to the parameter
            set the random state to 5

        Fit the model, and return the predictions and all classes prediction probabilities

        Args:
            X_train - array of the dataset of n sample points each with d features
            y_train - array of the label values for each sample point
            C - a scalar of the regularization hyperparameter
            X_test - array of the test dataset
        Returns:
            y_predict - array of the predicted classes
            y_prob - array of the prediction probability for each class
    """
    # Write your code here
    model_logisticRegression = LogisticRegression( multi_class='multinomial',solver='lbfgs' ,random_state=5, C=C)
    model_logisticRegression.fit(X_train,y_train)
    y_pred = model_logisticRegression.predict(X_test)
    y_prob = model_logisticRegression.predict_proba(X_test)
    return y_pred,y_prob

    
    


def softmax_prediction_and_probability_on_iris_dataset():
    iris = datasets.load_iris()  # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]           # 0 : setosa, 1 : versicolor, 2 : virginica
    C = 10
    X_test = [[4.0, 1.5], [1.0, 0.5], [6.0, 2.5]]
    y_predict, y_prob = pred_prob_softmax_reg(X, y, C, X_test)
    print(y_predict)
    print(y_prob)


# # a function to plot the decision boundaries
# def plot_softmax_iris_dataset():
#     iris = datasets.load_iris()
#     X = iris["data"][:, (2, 3)]
#     y = iris["target"]
#     C = 10
#     x0, x1 = np.meshgrid(
#         np.linspace(0, 8, 500).reshape(-1, 1),
#         np.linspace(0, 3.5, 200).reshape(-1, 1),
#     )
#     X_new = np.c_[x0.ravel(), x1.ravel()]
#     y_predict, y_prob = pred_prob_softmax_reg(X, y, C, X_new)
#
#     zz1 = y_prob[:, 1].reshape(x0.shape)
#     zz = y_predict.reshape(x0.shape)
#     plt.figure(figsize=(10, 4))
#     plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
#     plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
#     plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")
#     custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
#     plt.contourf(x0, x1, zz, cmap=custom_cmap)
#     contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
#     plt.clabel(contour, inline=1, fontsize=12)
#     plt.xlabel("Petal length", fontsize=14)
#     plt.ylabel("Petal width", fontsize=14)
#     plt.legend(loc="center left", fontsize=14)
#     plt.axis([0, 7, 0, 3.5])
#     plt.show()


def main():
    softmax_prediction_and_probability_on_iris_dataset()
    # plot_softmax_iris_dataset()


if __name__ == "__main__":
    main()
