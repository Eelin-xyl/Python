# Support Vector Machine (SVM)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\CW3\\playerdata.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# 测试
y_pred = classifier.predict(X_test)
print(round(classifier.score(X_test, y_test),4)*100, '%')

# 计算正确值
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.imshow(cm, cmap=plt.cm.Blues)
indices = range(len(cm))
plt.xticks(indices, set(y_pred))
plt.yticks(indices, set(y_pred))
plt.colorbar()
plt.xlabel('y_train')
plt.ylabel('y_pred')
for first_index in range(len(cm)):
    for second_index in range(len(cm[first_index])):
        plt.text(first_index, second_index, cm[first_index][second_index])
plt.show()