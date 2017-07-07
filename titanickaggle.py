# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:29:09 2017

@author: Aniket Pant
"""

# Titanic Dataset Analysis, ML Approach

# Importing General Modules
import pandas as pd
import seaborn as sns
import numpy as np

# Extracting data from CSV files
trainData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\train.csv', index_col=0)
testData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\test.csv', index_col=0)

'''
trainData.drop('Name', axis=1, inplace=True)
trainData.drop('Ticket', axis=1, inplace=True)
trainData.drop('Cabin', axis=1, inplace=True)
testData.drop('Name', axis=1, inplace=True)
testData.drop('Ticket', axis=1, inplace=True)
testData.drop('Cabin', axis=1, inplace=True)
'''

X = trainData.drop(['Survived', 'Name', 'Embarked', 'Ticket', 'Cabin', 'Sex', 'Age'], axis=1)
y = trainData['Survived']

'''
# create a Python list of feature names
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]
y = data["Survived"]
'''

# Importing KNN
from sklearn.neighbors import KNeighborsClassifier

# Generating KNN Instance
knn = KNeighborsClassifier(n_neighbors=1) # K-value = 1 (Optimize Later)

# Fitting data to model
knn.fit(X, y)
z = knn.predict([[3,1,0,7.25], [3,1,0,71.3]])
print(z)


'''
# Importing Train_Test_Split to generate training and testing set to check coefficients
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# Importing LinearReg Model in SciKit
from sklearn.linear_model import LinearRegression

# Generating Instance
linreg = LinearRegression()

# fit the model to the training data
linreg.fit(X_train, y_train)
'''