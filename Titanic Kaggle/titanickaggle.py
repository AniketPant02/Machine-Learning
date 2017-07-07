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

# Reassigning Sex Column Values -> 0 = Female, 1 = Male
trainData['Sex'].replace('male', 1, inplace=True)
trainData['Sex'].replace('female', 0, inplace=True)
# Reassigning Embarked Column Values -> 0 = C, 1 = S, 2 = Q
trainData['Embarked'].replace('C', 0, inplace=True)
trainData['Embarked'].replace('S', 1, inplace=True)
trainData['Embarked'].replace('Q', 2, inplace=True)

X = trainData.drop(['Survived', 'Name', 'Embarked', 'Ticket', 'Cabin', 'Age'], axis=1)
y = trainData['Survived']

# Importing KNN
from sklearn.neighbors import KNeighborsClassifier

# Generating KNN Instance
knn = KNeighborsClassifier(n_neighbors=9)

# Fitting data to model
knn.fit(X, y)

# Searching for an optimal value of K
from sklearn.cross_validation import cross_val_score

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print (scores.mean())