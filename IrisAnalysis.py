# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 19:31:06 2017

@author: Aniket Pant
"""

import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier

# Generating Instance of KNN model
knn = KNeighborsClassifier(n_neighbors=1)

# Fitting data to model
knn.fit(X, y)

z = knn.predict([[3,5,4,2]])

print (z)