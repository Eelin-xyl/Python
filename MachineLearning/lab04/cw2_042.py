import numpy as np

def regular_linear_closed_form(X, y, lambda_reg):
    """
    Computes the closed form solution of least-squares linear regression with l2 regularization

    Args:
        X - (n, d + 1) numpy matrix of the design matrix (n sample points, d features plus fictitious dimension)
        y - (n, ) numpy array of the label values for each sample point
        lambda_reg - a scalar of the regularization hyperparameter
    Returns:
        w - (d + 1, ) numpy array of the weights of linear regression with the offset in the last dimension
    """
    # Write your code here
    ans = np.linalg.inv(np.dot(X.T, X) + lambda_reg * np.identity(len(X[0])))
    ans = np.dot(ans, X.T)
    return np.dot(ans, y)
    

    
def check_regular_linear_closed_form():
    X = np.array([[0.25, 0.31, 1],
                  [0.07, 0.24, 1],
                  [0.65, 0.38, 1],
                  [0.13, 0.82, 1]])
    y = np.array([0.56, 0.38, 0.89, 0.23])
    lambda_reg = 0.5

    print(regular_linear_closed_form(X, y, lambda_reg))


def main():
    check_regular_linear_closed_form()


if __name__ == "__main__":
    main()
