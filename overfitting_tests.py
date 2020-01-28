# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Go through a pandas data frame (data) and fit the data column, y_data_column to
the data columns (data_columns)
"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
from itertools import combinations
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn import linear_model

# plt.xkcd()

# data = pd.read_csv('mango_chloro_refl3.csv')
data = pd.read_csv('flouro_mango_leaves.csv')
data = data.loc[(data['integration time'] == 250)]
data = data.loc[(data['LED'] == 'White LED')]
data = data.reset_index(drop=True)
# data_columns = ['450 nm', '500 nm', '550 nm', '570 nm', '600 nm', '650 nm']
# data_columns = ['450 nm', '500 nm', '550 nm']
data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)

print(data_columns)
data_columns = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm', '560 nm', '585 nm', '610 nm', '645 nm', '680 nm', '810 nm', '860 nm']

y_data = data['Total Chlorophyll (ug/ml)']
y_data = 1 / y_data
pd.set_option('display.max_columns', 500)
print(data)


class ModelFit(object):
    def __init__(self):
        self.wavelengths = None
        self.test_score = None
        self.test_stdev = None
        self.train_score = None
        self.train_stdev = None


_linear_model = linear_model.LinearRegression()
R2 = dict()
for i in range(1, len(data_columns)+1):
    print(i)
    R2[i] = ModelFit()
    for data_list in combinations(data_columns, i):
        number_params = len(data_list)
        print('num_ params = ', number_params)
        print(data_list)
        # fit all the data in data list using K-Fold cross validation
        cv_splitter = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
        # cv_splitter.get_n_splits(data)'
        j = 1
        x_data = data[list(data_list)]
        # print(x_data)
        R2_test_scores = []
        R2_train_scrores = []
        for train_index, test_index in cv_splitter.split(data):
            # print('=====', j)
            j += 1
            # print(train_index)
            # print(test_index)
            x_train, x_test = x_data.loc[train_index], x_data.loc[test_index]
            y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
            # print('X train')
            # print(x_train)
            # print('Y train')
            # print(y_train)
            model_fit = _linear_model.fit(x_train, y_train)
            test_prediction = _linear_model.predict(x_test)
            R2_test_scores.append(model_fit.score(x_test, y_test))
            R2_train_scrores.append(model_fit.score(x_train, y_train))
        R2_test_scores = np.array(R2_test_scores)
        R2_train_scrores = np.array(R2_train_scrores)
        # R2[i].append((R2_run_scores.mean(), R2_run_scores.std(), data_list))
        R2[i].wavelengths = data_list
        R2[i].test_score = R2_test_scores.mean()
        R2[i].test_stdev = R2_test_scores.std()
        R2[i].train_score = R2_train_scrores.mean()
        R2[i].train_stdev = R2_train_scrores.std()


print(R2)
print(R2.keys())
train_sizes = []
y_test_scores = []
y_test_std = []
y_train_scores = []
y_train_std = []
for free_parameters, model_details in R2.items():
    print('params =', free_parameters)
    print(model_details)
    print(model_details.test_score)
    best_test_value = 0
    best_details = None
    if model_details.test_score > best_test_value:
        best_test_value = model_details.test_score
        best_details = model_details
    print('best score: {0} | details: {1}'.format(best_test_value, best_details))

    y_test_scores.append(best_details.test_score)
    y_test_std.append(best_details.test_stdev)

    y_train_scores.append(best_details.train_score)
    y_train_std.append(best_details.train_stdev)

    train_sizes.append(free_parameters)

train_sizes = np.array(train_sizes)
y_test_scores = np.array(y_test_scores)
y_test_std = np.array(y_test_std)
y_train_scores = np.array(y_train_scores)
y_train_std = np.array(y_train_std)

# plt.plot(x_plt, y_plt)
plt.fill_between(train_sizes, y_train_scores - y_train_std,
                 y_train_scores + y_train_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, y_test_scores - y_test_std,
                 y_test_scores + y_test_std, alpha=0.1, color="g")
plt.plot(train_sizes, y_train_scores, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, y_test_scores, 'o-', color="g",
         label="Cross-validation score")

plt.title('Validation Curves')
plt.xlabel("Number of Model Parameters")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()

