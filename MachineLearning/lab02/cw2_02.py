import numpy as np


def perceptron(feature_matrix, labels, T):
    """
    Runs the perceptron algorithm on a dataset for T iterations with learning rate = 1.
    Assume the sample points are already in random order.

    Args:
        feature_matrix - A numpy matrix of the input dataset, where
            each row represents a single sample point.
        labels - A numpy array where the i-th element of the array is the
            label of the i-th row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of w,
        and the second element is a real number with the value of alpha.
    """
    # Write your code here
    w = np.zeros(feature_matrix[0].shape)
    a = 0
    lr = 1
    for _ in range(T):
        for i in range(len(feature_matrix)):
            if labels[i] * (np.dot(feature_matrix[i], w) + a) <= 0:
                w += lr * labels[i] * feature_matrix[i]
                a += lr * labels[i]
    return w, a

    
    


def average_hinge_loss(feature_matrix, labels, w, alpha):
    """
    Finds the average hinge loss on a dataset given linear classifier parameters.

    Args:
        feature_matrix - A numpy matrix of the input dataset, where
            each row represents a single sample point.
        labels - A numpy array where the i-th element of the array is the
            label of the i-th row of the feature matrix.
        w - A numpy array of the linear classifier weight vector parameter.
        alpha - A real number of the offset parameter.

    Returns: A real number representing the average hinge loss across all of
        the sample points in the feature matrix.
    """

    # Write your code here
    hl = []
    for i in range(len(feature_matrix)):
        z = np.dot(labels[i], (np.dot(w, feature_matrix[i]) + alpha))
        if z >= 1:
            hl.append(0)
        else:
            hl.append(1 - z)
    return sum(hl) / len(hl)

    
    
    


def check_perceptron():
    feature_matrix = np.array([[-1.0, -1.0], [1.0, 0.0], [-1.0, 10]])
    labels = np.array([1.0, -1.0, 1.0])
    T = 2
    feature_matrix = np.array([[-4,2],[-2,1],[-1,-1],[2,2],[1,-2]])
    labels = np.array([1.0,1.0,-1.0,-1.0,-1.0])
    T = 2
    print(perceptron(feature_matrix, labels, T))


def check_average_hinge_loss():
    feature_matrix = np.array([[1.0, 2.0], [5.0, 2.5]])
    labels = np.array([1.0, -1.0])
    w = np.array([1.0, 2.0])
    alpha = 1.5
    print(average_hinge_loss(feature_matrix, labels, w, alpha))


def main():
    check_perceptron()
    check_average_hinge_loss()


if __name__ == "__main__":
    main()