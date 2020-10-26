import numpy as np
from sklearn.svm import LinearSVC


def linear_svm(train_X, train_y, valid_X):
    """
    Trains and validates a linear SVM for binary classification

    Args:
        train_X - (n, d) numpy matrix of n sample points each with d features
        train_y - (n, ) numpy array containing the labels (0 or 1) for each training sample point
        valid_X - (m, d) numpy matrix of m sample points each with d features
    Returns:
        pred_valid_y - (m,) numpy array containing labels (0 or 1) for each validation sample point
    """
    # Write your code here
    classifier = LinearSVC(C=0.1, random_state=0)
    classifier.fit(train_X, train_y)
    y_pred = classifier.predict(valid_X)
    return y_pred
    


def check_linear_svm():
    train_X = np.array([[0.54447421, 0.18215857, 0.42080593, 0.09503258],
                        [0.07029242, 0.78303248, 0.49830402, 0.58931291],
                        [0.90763019, 0.23140388, 0.55020372, 0.94464876],
                        [0.02275208, 0.41993050, 0.77688928, 0.23471082],
                        [0.44327947, 0.46668775, 0.49620036, 0.8741674 ],
                        [0.89992171, 0.51378717, 0.93054792, 0.36577948],
                        [0.10143805, 0.92885263, 0.90649271, 0.03073832],
                        [0.23980516, 0.05080018, 0.36151902, 0.42582652],
                        [0.24875678, 0.31790184, 0.00244545, 0.04698004],
                        [0.91039726, 0.22337786, 0.88081661, 0.65443671],
                        [0.65780464, 0.10148759, 0.61612352, 0.09778783],
                        [0.34932217, 0.62225327, 0.26841827, 0.75045832],
                        [0.89249802, 0.15246971, 0.68264018, 0.73206384]])

    train_y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1])

    valid_X = np.array([[0.84001386, 0.2450525,  0.26251709, 0.60305993],
                        [0.73693401, 0.23732597, 0.65880612, 0.92740201],
                        [0.73735164, 0.92275345, 0.74036024, 0.52329581],
                        [0.6561221,  0.29895909, 0.20541315, 0.79888133],
                        [0.64286899, 0.12565544, 0.92603579, 0.42034257],
                        [0.17879218, 0.34025884, 0.97226287, 0.24545452],
                        [0.14002065, 0.37940135, 0.0618589,  0.62764347],
                        [0.80016359, 0.47420654, 0.69687303, 0.44035293],
                        [0.45199603, 0.28817775, 0.59474749, 0.99536194],
                        [0.88882929, 0.71100793, 0.6564737,  0.76430291],
                        [0.74766776, 0.34262488, 0.9329397,  0.98046449],
                        [0.83531797, 0.63329506, 0.56327902, 0.02033087]])

    print(linear_svm(train_X, train_y, valid_X))


def main():
    check_linear_svm()


if __name__ == "__main__":
    main()
