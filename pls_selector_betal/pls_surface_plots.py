# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import pickle
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from scipy.signal import savgol_filter

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class ModelFit(object):
    def __init__(self):
        self.test_score = []
        self.test_stdev = []
        self.train_score = []
        self.train_stdev = []

filename = 'ch_a_param_selector_16.pickle'
files = [f for f in os.listdir('.') if (os.path.isfile(f) and '.pickle' in f and 'total' in f)]



with open(filename, 'rb') as f:
    full_data_dict = pickle.load(f)

start = True
x = None
y = None
z = None
num_vars = 60
for i in range(1, num_vars):
    filename = 'ch_a_param_selector_{0}.pickle'.format(i)
    print(filename)
    with open(filename, 'rb') as f:
        data_slice = pickle.load(f)
        data_scores = data_slice['r2']
        number_variables = np.flip(np.array(data_slice['num variables']))
        scores = np.flip(np.array(data_scores.test_score))
        num_conditions = len(number_variables)
        if start:
            x = np.full((num_conditions, num_vars), np.nan)
            y = np.full((num_conditions, num_vars), np.nan)
            z = np.full((num_conditions, num_vars), np.nan)
            start = False

        x[i-1:, i-1] = number_variables
        y[i-1:, i-1] = i
        z[i-1:, i-1] = scores


        # test_scores = np.flip(np.array(data_scores.test_score))
        # print(test_scores)
        # print(num_conditions)
        # print(np.full(num_conditions, i))
        print('==========')
        print(z)

x = np.array(x)
y = np.array(y)
z = np.array(z)
# data_dict = full_data_dict['mae']

# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# print(z)
#
# surf = ax.plot_surface(x, y, z)
# im = plt.imshow(z, cmap='jet')
plt.pcolor(z, vmax=.996, vmin=0.98, cmap='jet')
plt.colorbar()
plt.show()

# x number of variables
# y number PLS components
# z r2 or mea score

