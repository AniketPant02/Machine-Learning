import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

trainData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\train.csv')
testData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\test.csv')

combineData = [trainData, testData]

# Making new column filled with NaN values
trainData['familySize'] = float('NaN')
testData['familySize'] = float('NaN')

# Fill in data using fillna
for dataframe in combineData:
    # Filling in embarked value with most prevalent port
    dataframe['Embarked'] = dataframe['Embarked'].fillna("S")
    # Filling in each NaN age value with median of feature
    dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].median())

# Only apparent in Test Dataframe. Filling NaN values with median of Fare values
testData.Fare = testData.Fare.fillna(testData['Fare'].median())

# Iterating through all Sex and Embarked entries in all dataframes inside "combineData" and remapping them
for dataframe in combineData:
    # Convert the male and female groups to integer form
    dataframe['Sex'] = dataframe['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    # Convert the Embarked classes to integer form
    dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
# Feature Engineering
# Creating Family Size Feature. +1 for including self
trainData['familySize'] = trainData['SibSp'] + trainData['Parch'] + 1
testData['familySize'] = testData['SibSp'] + testData['Parch'] + 1

# Making Feature Matrix and Observation Vector
feature_cols = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "familySize"]
X = trainData.loc[:, feature_cols]
y = trainData.Survived
X_test = testData.loc[:, feature_cols]

# Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
# Generating Model Instance
forestModel = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 60, random_state = 1)
randomForest = forestModel.fit(X, y)
print(randomForest.score(X, y))
test_Predictions = randomForest.predict(X_test)

print(test_Predictions)
pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':test_Predictions}).set_index('PassengerId').to_csv('submission.csv')