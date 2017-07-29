# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:14:28 2017

@author: Aniket Pant
"""

import numpy as np
import pandas as pd
x = np.random.randn(500)
y = np.sin(x)
df = pd.DataFrame({'x':x, 'y':y})
df.plot('x', 'y', kind='scatter')