# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import pickle
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
num_components = 52

class ModelFit(object):
    def __init__(self):
        self.test_score = []
        self.test_stdev = []
        self.train_score = []
        self.train_stdev = []

filename = "ch_a_param_selector_{0}.pickle".format(num_components)
# filename = "ch_a_param_selector_{0}.pickle".format(num_components)
with open(filename, 'rb') as f:
    full_data_dict = pickle.load(f)

print(full_data_dict)
print(full_data_dict.keys())

for key, value in full_data_dict.items():
    print(key, value)

data_dict = full_data_dict['r2']
number_pts = 200
# number_pts = 1000

test_scores_mean = np.flip(np.array(data_dict.test_score))[:number_pts]
test_scores_std = np.flip(np.array(data_dict.test_stdev))[:number_pts]
train_scores_mean = np.flip(np.array(data_dict.train_score))[:number_pts]
train_scores_std = np.flip(np.array(data_dict.train_stdev))[:number_pts]

train_sizes = np.flip(np.array(full_data_dict['num variables']))[:number_pts]
print(len(test_scores_mean), len(train_sizes))
# print(filename.split('_')[2].split('.'))
# num_components = int(filename.split('_')[2].split('.')[0])
print('num components =', num_components)
# train_sizes = np.linspace(num_components, num_components+number_pts, number_pts)
print(train_sizes)

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, '-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, '-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.xlabel("Number of variables in PLS")
plt.ylabel("R^2")
plt.title("{0} Component PLS".format(num_components))
plt.show()
