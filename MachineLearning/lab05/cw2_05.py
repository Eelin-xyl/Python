import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


# CW2 5.1
def cross_validate_svm(X, y, alphas):
    """
        Compute the mean cross validation scores over a number of hyperparameter alphas.
        As a model, use SGDClassifier that applies Stochastic Gradient Descent to train a linear SVM.
        Set its loss function to hinge loss function, and its penalty function to L2-norm penalty function.
        In addition, set 5 as its random state.
        Use StandardScaler to scale the dataset, combine it with the model using a pipeline.
        Use 5-fold cross-validation and the pipeline to obtain the validation score
            for each value of alphas (= lambdas in lecture).
        Store the mean of five scores for each alpha in a numpy array and return it.

        Args:
            X - (n, d) numpy matrix of n sample points each with d features
            y - (n, ) numpy array containing the labels (0 or 1) for each sample point
            alphas - (k, ) numpy array of k candidate hyperparameter alphas (= lambdas in lecture)
        Returns:
            val_scores - (k, ) numpy array containing mean cross validation scores of each of the k alphas
    """
    # Write your code here
    ans = []
    for i in alphas:
        pipe = Pipeline(steps = [('scaler', StandardScaler()),
                                 ('classifier', SGDClassifier(loss = 'hinge',penalty = 'L2', random_state = 5, alpha = i))])
        scores = cross_val_score(pipe, X, y, cv = 5)
        ans.append(scores.mean())
    return  ans
    
    


# CW2 5.2
def train_svm_alpha_star(X, y, alpha_star):
    """
        Train an SVM with alpha_star and compute the training accuracy.
        As a model, use SGDClassifier that applies Stochastic Gradient Descent to train a linear SVM.
        Set its loss function to hinge loss function, and its penalty function to L2-norm penalty function.
        In addition, set 5 as its random state, and alpha_star as its regularization hyperparameter.
        Use StandardScaler to scale the dataset.
        Compute the training accuracy score on the scaled dataset.

        Args:
            X - (n, d) numpy matrix of n sample points each with d features
            y - (n, ) numpy array containing the labels (0 or 1) for each sample point
            alpha_star - a scalar of the hyperparameter alpha (= lambda in lecture)
        Returns:
            train_accuracy - a scalar of the accuracy score of the model on the scaled training data
    """
    # Write your code here
    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                           ('classifier', SGDClassifier(loss='hinge', penalty='L2', random_state=5, alpha=alpha_star))])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
   # print(y_pred)
    a = 0
    b = 0
    for i in y_pred:
        if i == y[b]:
            a = a + 1
        b = b + 1
    return a / b
    


# def cross_validation_svm_on_breast_cancer_dataset():
#     cancer_dataset = datasets.load_breast_cancer()  # https://scikit-learn.org/stable/datasets/index.html#breast-cancer-dataset
#     X = cancer_dataset.data                         # 30 numerical features
#     y = cancer_dataset.target                       # labels: malignant = 0, benign = 1

#     alphas = np.array([0.01, 0.05, 0.10, 0.5, 1.0])
#     val_scores = cross_validate_svm(X, y, alphas)

#     # Determine the alpha that maximizes the cross-validation score
#     index = np.argmax(val_scores)
#     alpha_star = alphas[index]
#     print('alpha_star =', alpha_star)

#     train_accuracy = train_svm_alpha_star(X, y, alpha_star)
#     print('train_accuracy =', train_accuracy)


# def main():
#     cross_validation_svm_on_breast_cancer_dataset()


# if __name__ == "__main__":
#     main()

