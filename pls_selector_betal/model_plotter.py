# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import pickle
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

num_components = 39
plt.style.use('seaborn')


def calc_leds(data):
    print(data.columns)
    leds = []
    for column in data.columns:
        print(column)
        led = column.split(',')[1].strip()
        print(led)
        if led not in leds:
            leds.append(led)

    print(leds)
    print(len(leds))

def remove_led(data, led):
    columns_to_keep = []
    for column in data.columns:
        if led not in column:
            columns_to_keep.append(column)
    # print(data[columns_to_keep])
    return data[columns_to_keep]

class ModelFit(object):
    def __init__(self):
        self.test_score = []
        self.test_stdev = []
        self.train_score = []
        self.train_stdev = []

data_full = pd.read_csv('wildbetal_flouro_rows.csv')
# data_full = data_full[data_full['position'] == 'pos 2']
y_name = 'Total Chlorophyll (ug/ml)'
y_data = data_full[y_name]


# y_data = data_full['Chlorophyll a (ug/ml)'] / data_full['Chlorophyll b (ug/ml)']

# print(data_full)
x_data_columns = []
for column in data_full:
    if 'nm' in column:
        x_data_columns.append(column)

x_data = data_full[x_data_columns]
x_data_scaled = StandardScaler().fit_transform(x_data)
x_data = pd.DataFrame(x_data_scaled, columns=x_data.columns)

filename = "total_param_selector_{0}_m.pickle".format(num_components)
# filename = "ch_a_param_selector_{0}.pickle".format(num_components)
with open(filename, 'rb') as f:
    data_summary = pickle.load(f)

print(data_summary)
print(data_summary.keys())
print('========')
for key, value in data_summary.items():
    print(key, value)

print('=====+++')
print(data_summary['columns'].values)
# print(data_summary.columns)
print(x_data)
x = x_data[data_summary['columns'].values]
print(x)
# x = remove_led(x, '390 nm LED')
# x = remove_led(x, '455 nm LED')
# x = remove_led(x, '475 nm LED')

pls = PLSRegression(num_components)

cv_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25)
# cv_splitter = ShuffleSplit(n_splits=1, test_size=0.35)
group_splitter = data_full['Leaf number']

for train_index, test_index in cv_splitter.split(x_data, groups=group_splitter):
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]

    model_fit = pls.fit(x_train, y_train)
    y_pred_train = pls.predict(x_train)
    y_pred_test = pls.predict(x_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # print(x_train)
    # print(x_train.columns)
    # print(x_train.index)
    # print(x_test.index)
    # print(r2_train, r2_test)
    # print(x.columns)
    # print(x)
    # new_x = remove_led(x, '390 nm LED')
    # print(new_x)
    # calc_leds(x)
    plt.scatter(y_train, y_pred_train, color='red', label="Training set")
    plt.scatter(y_test, y_pred_test, color='blue', label="Test set")
    plt.annotate("R\u00B2 training: {:.3f}".format(r2_train), xy=(0.17, 0.5))
    plt.annotate("R\u00B2 test: {:.3f}".format(r2_test), xy=(0.17, 0.45))
    plt.xlabel(y_name)
    plt.ylabel("Predicted {0}".format(y_name))
    plt.legend()
    plt.title("Mange AS7265x device\nPLS with {0} components".format(num_components))
    plt.show()
