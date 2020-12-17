# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned3).csv', encoding='gbk')

dataset=dataset.drop('key',axis=1)
dataset=dataset.drop('artist_name',axis=1)

dataset = dataset.iloc[:,:].values

X = dataset[:, 1:]
y = dataset[:, 0]

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer([('artist_name', OneHotEncoder(), [0])], remainder = 'passthrough')
# X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(round(classifier.score(X_test, y_test),4)*100, '%')

# Making the Confusion Matrix
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