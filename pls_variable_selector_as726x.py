# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Variable Selection with PLS Regression for Small Data Set

Script and functions to go threw a small data set, i.e. 6 spectral channels
of an AS7262, and fit a PLS Regression with 1-# spectral channels components,
and for each n_components PLS Regression go through and find the best
varaialbes
"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

# local files
import data_getter

plt.style.use('ggplot')

# Prepare data
spectrum_data, chloro_data, full_data = data_getter.get_data("as7262 mango")

print(spectrum_data.columns)
print(chloro_data.columns)
print(full_data.columns)

class ModelFit(object):
    def __init__(self):
        self.test_score = []
        self.test_stdev = []
        self.train_score = []
        self.train_stdev = []

    def add_data(self, data):
        self.test_score.append()


def variable_selector(_estimator, _x, _y, min_components,
                      _cv=None, _cv_groups=None,
                      _scorer=None):

    for i in range(min_components, _x.shape[1], 1):
        print(i, _x.shape)
        _estimator.fit(_x, _y)

        y_cv = cross_validate(_estimator, _x, _y,
                              cv=_cv, groups=_cv_groups,
                              scoring=_scorer,
                              return_train_score=True)
        y_cv_test_mean = y_cv['test_score'].mean()
        y_cv_test_std = y_cv['test_score'].std()
        y_cv_train_mean = y_cv['train_score'].mean()
        y_cv_train_std = y_cv['train_score'].std()
        print(y_cv_test_mean, y_cv_test_std, y_cv_train_mean, y_cv_train_std)



if __name__ == '__main__':
    num_components = 3
    pls = PLSRegression(n_components=num_components)
    y_name = 'Total Chlorophyll (ug/ml)'
    cv = ShuffleSplit(n_splits=100)
    variable_selector(pls, spectrum_data, chloro_data[y_name],
                      num_components,
                      _cv=cv, _scorer='r2')
