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
feature_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']

X = trainData.loc[:, feature_cols]
y = trainData.Survived

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X, y)

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

'''
X_test = testData.loc[:, feature_cols]
X_test = X_test.fillna(method="ffill", limit=2)
test_Predictions = knn.predict(X_test)

pd.DataFrame({'PassengerId':testData.PassengerId, 'Survived':test_Predictions}).set_index('PassengerId').to_csv('submission.csv')
'''