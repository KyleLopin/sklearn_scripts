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

filename = 'total_param_selector_16.pickle'
files = [f for f in os.listdir('.') if (os.path.isfile(f) and '.pickle' in f and 'total' in f)]

print(files)

with open(filename, 'rb') as f:
    full_data_dict = pickle.load(f)

# data_dict = full_data_dict['mae']
num_pts = 200
test_scores_mean_mae = np.flip(np.array(full_data_dict['mae'].test_score))[:num_pts]
test_scores_mean_r2 = np.flip(np.array(full_data_dict['r2'].test_score))[:num_pts]
train_sizes = np.flip(np.array(full_data_dict['num variables']))[:num_pts]
y = test_scores_mean_mae

y_filter_der = savgol_filter(y, 21, polyorder=3, deriv=1)
y_filter = savgol_filter(y, 21, polyorder=3)
y_zero_index = np.where(y_filter_der < 0)[0][0]

print(y_zero_index)
print(train_sizes[y_zero_index])
# print(train_sizes)
# # print(test_scores_mean.shape)
# print(train_sizes.shape)
# x = np.linspace(1, 40, 40)
# print(x)

# y = (1 - np.exp(-2*x)) * np.exp(-x/10)
x = train_sizes

# print(x)
x = x[:, np.newaxis]
# print(x)
model = make_pipeline(PolynomialFeatures(5), Ridge())
model.fit(x, y)
y_plot = model.predict(x)

# plt.plot(x, y)
plt.grid()
plt.plot(x, y)
# plt.plot(x, y_filter)
plt.plot(x, y_plot, 'r')
# plt.plot(train_sizes, test_scores_mean_r2)
# plt.show()

# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# X, Y = np.meshgrid(x, y)
# Z = X*np.exp(-X - Y)
#
# print(X)
# print('======')
# print(Y)
# print('======')
# print(Z)

# x number of variables
# y number PLS components
# z r2 or mea score

