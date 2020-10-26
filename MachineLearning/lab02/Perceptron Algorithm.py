import numpy as np

feature_matrix = np.array([[-1.0, -1.0], [1.0, 0.0], [-1.0, 10]])
labels = np.array([1.0, -1.0, 1.0])
T = 2

w = np.zeros(feature_matrix[0].shape)
a = 0
lr = 1
for _ in range(T):
    for i in range(3):
        if labels[i] * (np.dot(feature_matrix[i], w) + a) <= 0:
            w += lr * np.dot(labels[i], feature_matrix[i])
            a += a + lr * labels[i]

print(w, a)
# for _ in range(T):
#     for i in range(3):
#         if labels[i] * (np.dot(feature_matrix[i], w) + a) <= 0:
#             print(1)