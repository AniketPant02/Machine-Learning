# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:29:09 2017

@author: Aniket Pant
"""

# Titanic Dataset Analysis, ML Approach
    
# Importing General Modules
import pandas as pd
import seaborn as sns

# Extracting data from CSV files
trainData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\train.csv', index_col=0)
testData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\test.csv', index_col=0)
# Adding Survived Columns to Test Data CSV
# testData = testData.assign(Survived=pd.Series(np.random.randn(testDataLength)).values)
# Reassigning Sex Column Values -> 0 = Female, 1 = Male
trainData['Sex'].replace('male', 1, inplace=True)
trainData['Sex'].replace('female', 0, inplace=True)
# Reassigning Embarked Column Values -> 0 = C, 1 = S, 2 = Q
trainData['Embarked'].replace('C', 0, inplace=True)
trainData['Embarked'].replace('S', 1, inplace=True)
trainData['Embarked'].replace('Q', 2, inplace=True)
# Reassigning Column Values in Test CSV
testData['Sex'].replace('male', 1, inplace=True)
testData['Sex'].replace('female', 0, inplace=True)
testData['Embarked'].replace('C', 0, inplace=True)
testData['Embarked'].replace('S', 1, inplace=True)
testData['Embarked'].replace('Q', 2, inplace=True)


X_train = trainData.drop(['Survived', 'Name', 'Embarked', 'Ticket', 'Cabin', 'Age'], axis=1)
y_train = trainData['Survived']
X_test = testData.drop(['Survived', 'Name', 'Embarked', 'Ticket', 'Cabin', 'Age'], axis=1)
y_test = testData['Survived']

# Importing KNN
from sklearn.neighbors import KNeighborsClassifier

# Generating KNN Instance
knn = KNeighborsClassifier(n_neighbors=9)

# Fitting data to model
knn.fit(X_train, y_train)

submission = pd.DataFrame({
        "PassengerId": testData.index,
        "Survived": knn.predict(X_test, y_test)
    })

submission.to_csv('test.csv', index=False)