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

trainData.drop('Name', axis=1, inplace=True)
trainData.drop('Ticket', axis=1, inplace=True)
trainData.drop('Cabin', axis=1, inplace=True)
testData.drop('Name', axis=1, inplace=True)
testData.drop('Ticket', axis=1, inplace=True)
testData.drop('Cabin', axis=1, inplace=True)

sns.pairplot(trainData, x_vars=["Pclass", "Age", "SibSp", "Parch", "Fare"], y_vars=["Survived"], size=7, aspect=.8, kind="reg")
