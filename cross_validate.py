# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR

import statsmodels.api as sm


refl_data = pd.read_csv('mango_chloro_refl3.csv')


x = refl_data['550 nm'].values.reshape(-1, 1)
y = refl_data['Total Chlorophyll (ug/ml)'].values.reshape(-1, 1)
print(x)
print(y)
# refl_data = refl_data.loc[(refl_data['position'] == 'pos 3')]
refl_data = refl_data.groupby('Leaf number', as_index=True).mean()
train_sizes, train_scores, valid_scores = learning_curve(SVR(kernel='linear'), x, y)
# , train_sizes=[50, 80, 110], cv=5
print(train_sizes, train_scores)

