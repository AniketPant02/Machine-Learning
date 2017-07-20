# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:31:50 2017

@author: Aniket Pant
"""

import pandas as pd

trainData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\train.csv')
testData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Titanic Kaggle\\data\\test.csv')

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
trainData = trainData.fillna(method="ffill", limit=4)
feature_cols = ['Pclass', 'Sex', 'Parch', 'Fare']

X = trainData.loc[:, feature_cols]
y = trainData.Survived

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, y)


'''
from sklearn.cross_validation import cross_val_score
k_range = list(range(1, 31))
k_scores = []
for k in range(k_range):
    knn = KNeighborsClassifier(n_neighbors=k)
    scoresknn = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier(n_estimators=25)
randomForest.fit(X, y)

n_range = list(range(1, 31))
n_scores = []
for n in range(n_range):
    randomForest = RandomForestClassifier(n_estimators=n)
    scoresRandomForest = cross_val_score(randomForest, X, y, cv=10, scoring='accuracy')
    n_scores.append(scores.mean())
print(n_scores)
'''

X_test = testData.loc[:, feature_cols]
X_test = X_test.fillna(method="ffill", limit=2)
test_Predictions = knn.predict(X_test)

pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':test_Predictions}).set_index('PassengerId').to_csv('submission.csv')

'''
# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
'''
