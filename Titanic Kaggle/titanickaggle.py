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

# Reassigning Values
trainData.replace('male', 1)
trainData.replace('female', 0)


X = trainData.drop(['Survived', 'Name', 'Embarked', 'Ticket', 'Cabin', 'Sex', 'Age'], axis=1)
y = trainData['Survived']


'''
sns.pairplot(trainData, x_vars=['Pclass', 'SibSp', 'Parch', 'Fare'], y_vars=['Survived'], size=7, aspect=0.7,kind='reg')
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=trainData,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);
'''

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

'''
k_range = list(range(1,31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
plt.plot(k_range, k_scores)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.show()
'''
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