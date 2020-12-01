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

data = dataset.iloc[:, :].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder = LabelEncoder()
data[:,0] = labelEncoder.fit_transform(data[:,0])
data[:,1] = labelEncoder.fit_transform(data[:,1])

ct = ColumnTransformer([('key', OneHotEncoder(), [7])], remainder = 'passthrough')
data = ct.fit_transform(data)


data=data[:, 1:]


train_features = data[:8000,:-1]
test_features = data[8000:10000,:-1]
train_targets = data[:8000,-1]
test_targets = data[8000:10000,-1]




"""
Train the model
"""

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)


"""
Fit tree
"""
tree = tree.fit(train_features, train_targets)

"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)



"""
Check the accuracy
"""

print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")