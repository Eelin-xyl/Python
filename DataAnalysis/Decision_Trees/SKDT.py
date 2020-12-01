"""
Import the DecisionTreeClassifier model.
"""

#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


"""
Import the Zoo Dataset
"""

#Import the dataset 
dataset = pd.read_csv(r'Python\DataAnalysis\SpotifyFeatures(cleaned2).csv',encoding="gbk")
#We drop the animal names since this is not a good feature to split the data on
# dataset=dataset.drop('artist_name',axis=1)
# dataset=dataset.drop('track_name',axis=1)
# dataset=dataset.drop('key',axis=1)
dataset=dataset.drop('track_id',axis=1)




"""
Split the data into a training and a testing set
"""

X = dataset.iloc[:, :].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# X[:,7] = labelEncoder.fit_transform(X[:,7])

ct = ColumnTransformer([('key', OneHotEncoder(), [7])], remainder = 'passthrough')
X = ct.fit_transform(X)
# oneHotEncoder = OneHotEncoder(categories='key')

labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
X[:,1] = labelEncoder.fit_transform(X[:,1])
# X = oneHotEncoder.fit_transform(X).toarray()
# X=X[:, 1:]


train_features = X[:80,:-1]
test_features = X[80:100,:-1]
train_targets = dataset.iloc[:80,-1].values
test_targets = dataset.iloc[80:100,-1].values




"""
Train the model
"""

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)



"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)



"""
Check the accuracy
"""

print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")