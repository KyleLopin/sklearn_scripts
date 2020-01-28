# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
from itertools import combinations
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ShuffleSplit, GroupShuffleSplit
from sklearn import linear_model

# plt.xkcd()
plt.style.use('seaborn')

# data = pd.read_csv('mango_chloro_refl3.csv')

data_full = pd.read_csv('mango_flouro_rows.csv')
# data = data.loc[(data['integration time'] == 250)]
print(data_full.columns)


LEDs = ['White LED', 'IR LED', 'UV (405 nm) LED', '390 nm LED', '395 nm LED',
        '400 nm LED', '405 nm LED', '410 nm LED', '425 nm LED', '455 nm LED',
        '465 nm LED', '470 nm LED', '475 nm LED', '480 nm LED', '505 nm LED',
        '525 nm LED', '630 nm LED', '890 nm LED', '940 nm LED']
LED = LEDs[0]
data_columns = []

y_name = 'Chlorophyll a (ug/ml)'
y_data = data_full['Total Chlorophyll (ug/ml)']
# calculate ratio
# y_data = data_full['Chlorophyll a (ug/ml)'] / data_full['Chlorophyll b (ug/ml)']
print(y_data)

for column in data_full.columns:
    if LED in column:
        data_columns.append(column)

data = data_full[data_columns]
print(data.columns)

data = data.reset_index(drop=True)
# data_columns = ['450 nm', '500 nm', '550 nm', '570 nm', '600 nm', '650 nm']
# data_columns = ['450 nm', '500 nm', '550 nm']

for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)


x_data = data[data_columns]

print(x_data)

y_data = 1 / y_data
pd.set_option('display.max_columns', 500)

# cv_splitter = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
cv_splitter = GroupShuffleSplit(n_splits=100,
                                test_size=0.35,
                                random_state=120)
group_splitter = data_full['Leaf number']

class ModelFit(object):
    def __init__(self):
        self.wavelengths = None
        self.test_score = None
        self.test_stdev = None
        self.train_score = None
        self.train_stdev = None
        self.coeff = None

_linear_model = linear_model.LinearRegression()
R2 = dict()


for number_params in range(1, 20):
    pls = PLSRegression(n_components=number_params)
    R2[number_params] = ModelFit()

    print('num_ params = ', number_params)
    # fit all the data in data list using shuffle cross validation
    # cv_splitter = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
    # cv_splitter.get_n_splits(data)'
    j = 1
    R2_test_scores = []
    R2_train_scores = []
    for train_index, test_index in cv_splitter.split(x_data, y_data, group_splitter):
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
        model_fit = pls.fit(x_train, y_train)
        y_pred_test = pls.predict(x_test)
        r2_test = r2_score(y_test, y_pred_test)
        R2_test_scores.append(r2_test)

        y_pred_train = pls.predict(x_train)
        r2_train = r2_score(y_train, y_pred_train)
        R2_train_scores.append(r2_train)
        # print('r2_test = {0}| r2_train = {1}'.format(r2_test, r2_train))
        # test_prediction = _linear_model.predict(x_test)
        # R2_test_scores.append(model_fit.score(x_test, y_test))
        # R2_train_scores.append(model_fit.score(x_train, y_train))
        R2[number_params].coeff = pls.coef_
        # print(pls.coef_)
    R2_test_scores = np.array(R2_test_scores)
    R2_train_scores = np.array(R2_train_scores)
    # print(R2_test_scores)
    R2[number_params].test_score = R2_test_scores.mean()
    R2[number_params].test_stdev = R2_test_scores.std()
    R2[number_params].train_score = R2_train_scores.mean()
    R2[number_params].train_stdev = R2_train_scores.std()

    print('r2_test = {0}| r2_train = {1}| params = {2}'.format(R2[number_params].test_score,
                                                 R2_train_scores.mean(), number_params))
    # print(R2[1].test_score)
    # print(pls.x_scores_)

train_sizes = []
y_test_scores = []
y_test_std = []
y_train_scores = []
y_train_std = []
print(R2)
for free_parameters, model_details in R2.items():
    # print('params =', free_parameters)
    # print(model_details)
    # print(model_details.test_score)
    # print(R2[1])
    best_test_value = -1
    best_details = None
    # print(model_details.test_score, best_test_value, model_details.test_score > best_test_value)
    if model_details.test_score > best_test_value:
        best_test_value = model_details.test_score
        best_details = model_details
    print('best score: {0} | details: {1}'.format(best_test_value, best_details))
    # print(model_details.coeff)
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

plt.title('Validation Curves for {0} LED\nPartial Least square regression'.format(LED))
plt.xlabel("Number of Model Parameters")
plt.xticks([0, 3, 6, 9, 12, 15, 18])
plt.ylabel("Score")
plt.legend(loc="best")
print(y_name)
print(LED)
indexes = np.argsort(np.array(y_test_scores))
best_index = indexes[-1]

print(y_test_scores[best_index])
print(y_test_std[best_index])

plt.show()


