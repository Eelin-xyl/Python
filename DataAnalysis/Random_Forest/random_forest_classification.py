# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned3).csv', encoding='gbk')

# dataset=dataset.drop('artist_name',axis=1)
# dataset=dataset.drop('track_name',axis=1)
# dataset=dataset.drop('key',axis=1)
# dataset=dataset.drop('track_id',axis=1)

# dataset=dataset.drop('speechiness', axis=1)
# dataset=dataset.drop('loudness',axis=1)
# dataset=dataset.drop('valence',axis=1)
# dataset=dataset.drop('duration_ms',axis=1)
# dataset=dataset.drop('instrumentalness',axis=1)
# dataset=dataset.drop('acousticness',axis=1)
dataset=dataset.drop('tempo',axis=1)
# dataset=dataset.drop('liveness',axis=1)
dataset=dataset.drop('mode',axis=1)
dataset=dataset.drop('time_signature'  ,axis=1)


dataset = dataset.iloc[:,:].values

X = dataset[:, 1:]
y = dataset[:, 0]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('artist_name', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

print(X[:3, -3])
labelencoder_X = LabelEncoder()
X[:,-3]=labelencoder_X.fit_transform(X[:,-3])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = 'gini', n_estimators=50, class_weight=  'balanced_subsample', max_features='sqrt')

# classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=50, min_samples_split=10, 
# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
# min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
# random_state=None, verbose=0, warm_start=False, class_weight=None)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test)*100, '%')

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred, average='weighted'), 'precision wei')

from sklearn.metrics import recall_score
print(recall_score(y_test, y_pred, average='weighted'), 'recall wei')

from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='weighted'))

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