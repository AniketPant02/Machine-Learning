# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 05:31:48 2017

@author: Aniket Pant
"""
# Pandas is used for data exploration
import pandas as pd
import seaborn as sns

# read CSV file from URL and save as a variable
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.7,kind='reg')

feature_cols = ['TV', 'Radio', 'Newspaper']

X = data[feature_cols]

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Splitting data into testing and training sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Introducing Linear Regression into Model
from sklearn.linear_model import LinearRegression
#Initialize
linreg = LinearRegression()
#Fit the model to the training data (finding line of best fit)
linreg.fit(X_train, y_train)

print(linreg.intercept_)
print(linreg.coef_)

# Pair features with respective coefficents
featureCoef = list(zip(feature_cols, linreg.coef_))
print(featureCoef)