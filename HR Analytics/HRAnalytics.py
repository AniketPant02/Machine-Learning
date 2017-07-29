# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:46:22 2017

@author: Aniket Pant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\HR Analytics\\HR_comma_sep.csv')

feature_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']

X = trainData.loc[:, feature_cols]
y = trainData.left

sns.pairplot(trainData, x_vars=feature_cols, y_vars='left', size=7, aspect=0.7, kind='reg')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X, y)
