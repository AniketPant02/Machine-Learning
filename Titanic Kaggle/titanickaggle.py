# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:29:09 2017

@author: Aniket Pant
"""

# Titanic Dataset Analysis, ML Approach

# Importing General Modules
import pandas as pd
import numpy as np

# Extracting data from CSV format
data = pd.read_csv('https://www.kaggle.com/c/titanic/download/gender_submission.csv', index_col=0)
print (data)
