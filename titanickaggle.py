# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:29:09 2017

@author: Aniket Pant
"""

# Titanic Dataset Analysis, ML Approach

# Importing General Modules
import pandas as pd
import seaborn as sns

# Extracting data from CSV format
trainData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\train.csv')
testData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\test.csv')

# Dropping unnecessary columns not needed for data analysis
trainData = trainData.drop(['PassengerId','Name','Ticket'], axis=1)
testData    = testData.drop(['Name','Ticket'], axis=1)

# Introduce LinRegression here to see what affects data most (or use all fields?)
sns.pairplot(trainData, x_vars=['Sex', 'Age', 'SibSp', 'Parch', 'Fare'], y_vars='Survived', size=7, aspect=0.7,kind='reg')