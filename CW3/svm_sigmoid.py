# Support Vector Machine (SVM)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\CW3\\playerdata.csv')

dataset=dataset.drop('MIN',axis=1)
dataset=dataset.drop('2FGM',axis=1)
dataset=dataset.drop('2FGA',axis=1)
dataset=dataset.drop('OREB',axis=1)

dataset.drop(dataset[dataset['TO'] > 6].index, inplace=True)

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
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

# 测试
y_pred = classifier.predict(X_test)
print(round(classifier.score(X_test, y_test),4)*100, '%')

# 计算正确值
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.matshow(cm, cmap=plt.cm.Blues) 
plt.colorbar()
for i in range(len(cm)): 
    for j in range(len(cm)):
        plt.annotate(cm[i,j], xy=(i, j))
plt.ylabel('True label')
plt.xlabel('Predicted label') 
plt.show()