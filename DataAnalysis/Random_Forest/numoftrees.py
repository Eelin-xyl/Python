#Finding the optimum number of n_estimators

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned3).csv', encoding='gbk')


dataset=dataset.drop('key',axis=1)
dataset=dataset.drop('artist_name',axis=1)

dataset = dataset.iloc[:,:].values

X = dataset[:, 1:]
y = dataset[:, 0]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

list1 = []
for estimators in range(50,200,25):
    classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='gini')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(50,200,25)), list1)
plt.show()