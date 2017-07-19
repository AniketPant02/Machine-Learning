# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:53 2017

@author: Aniket Pant
"""

# Apple Model using Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

stockFrame = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\AppleModel\\aapl.csv')
# stockFrame = pd.to_datetime(stockFrame.Date)
feature_cols = ['Open', 'High', 'Low', 'Volume']

X = stockFrame.loc[:, feature_cols]
y = stockFrame.Close

# sns.pairplot(stockFrame, x_vars=['Close'], y_vars=['Open', 'High', 'Low', 'Volume'], size=7, aspect=0.7, kind='reg')

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=False)
model.fit(X,y)

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(scores)
