# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:14:28 2017

@author: Aniket Pant
"""

import numpy as np
from pandas import read_hdf
# this query selects the columns A and B# where the values of A is greather than 0.5
hdf = read_hdf('C:\\Users\\Aniket Pant\\Downloads\\inhibitors\\cdk2.h5')

# x = pd.read_hdf("C:\\Users\\Aniket Pant\\Downloads\\inhibitors\\cdk2.h5")
# x = HDFStore('C:\\Users\\Aniket Pant\\Downloads\\inhibitors\\cdk2.h5')