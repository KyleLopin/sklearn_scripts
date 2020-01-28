# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Functions to make data sets that are to be analyzed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import helper_functions as funcs

plt.style.use('seaborn')

__author__ = "Kyle Vitatus Lopin"


as7262_files = funcs.get_all_files_with_stub('as7262')
print(as7262_files)
background = [250271, 176275, 216334, 219763, 230788, 129603]


all_data = None
starting = True

for file in as7262_files:
    print(file)
    try:
        file_data = pd.read_excel(file, sheet_name="Sheet1")
    except:
        file_data = pd.read_excel(file, sheet_name="Sheet2")
    if starting:
        all_data = file_data
        starting = False
    else:
        print('appending')
        all_data = all_data.append(file_data)
        # print(all_data)

print('=========')
print(all_data)

average_data = all_data.groupby('Leaf number', as_index=True).mean()

print('++++++')
print(average_data)

chloro_data_filename = funcs.get_all_files_with_stub('absorbance')[0]
chlor_data = pd.read_excel(chloro_data_filename, sheet_name='Summary')
print(chlor_data)
# print(chlor_data.index)
# print(chlor_data)
# chlor_data['leaf number:'] = 'Leaf: ' + chlor_data['leaf number:'].astype(str)
# print(chlor_data.columns)
# # chlor_data.rename(columns={'leaf number:', 'Leaf number'}, inplace=True)
# # chlor_data.rename(columns={'Leaf number', 'leaf number:'}, inplace=True)
# chlor_data['Leaf number'] = chlor_data['leaf number:']
# del chlor_data['leaf number:']
# chlor_data.set_index('Leaf number', inplace=True)
# print(chlor_data)

# average_data.insert(len(average_data.columns),)
# total_data = average_data.merge(chlor_data, left_index=True, right_index=True)
# total_data

refl_data = pd.read_csv('mango_chloro_refl3.csv')
refl_data = refl_data.groupby('Leaf number', as_index=True).mean()
average_data = refl_data


print(chlor_data.columns)
for _data in chlor_data.columns:
    average_data[_data] = chlor_data[_data]
# pd.set_option('display.max_columns', 500)
print('===========')
print(average_data)
print(average_data.columns)

data_columns = ['450 nm', '500 nm', '550 nm',
                '570 nm', '600 nm', '650 nm']
chloro_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']


def reg_m(_y, _x):
    ones = np.ones(len(_x))
    _X = sm.add_constant(np.column_stack((_x, ones)))
    _results = sm.OLS(_y, _X).fit()
    return _results

best_r = 0
best_params = None
best_columns = None
y_name = 'Total Chlorophyll (ug/ml)'
best_i = None
best_j = None

for i, column_i in enumerate(data_columns):
    for j, column_j in enumerate(data_columns):
        # average_data.plot.scatter(data_column, 'Total Chlorophyll (ug/ml)')
        y = average_data[y_name]
        # y = np.log(1/y)
        y = 1/ y
        results = reg_m(y, average_data[[column_i, column_j]])

        r_sq_adj = results.rsquared
        print(r_sq_adj)

        if r_sq_adj > best_r:
            best_r = r_sq_adj
            best_params = results.params
            best_columns = [column_i, column_j]
            best_i = i
            best_j = j
        # fitted_x = (average_data[column_i] * results.params.x1 +
        #             average_data[column_j] * results.params.x2 +
        #             results.params.const)
        # fitted_y = results.predict()
        #
        # plt.plot(x, fitted_y)
        # plt.annotate("R_adj**2 =\n{:.3f}".format(r_sq_adj), xy=(0.8, 0.8),
        #              xycoords='axes fraction')
        # plt.xlabel("{0} nm counts".format(data_column.split('.')[0]))
plt.show()
print('best r =', best_r)
print(best_params, best_columns)

print(y, best_columns)
results = reg_m(y, average_data[best_columns])

print(results.summary())
fitted_x = (results.params.x1 * average_data[best_columns[0]] +
            results.params.x2 * average_data[best_columns[1]] +
            results.params.const)

fitted_y = results.predict()

plt.plot(fitted_x, y, 'o', color='slateblue')
plt.plot(fitted_x, fitted_y, 'orangered')
plt.title("Linear Model Fit", size=22)
plt.annotate(u"R_adj\u00B2 =\n{:.3f}".format(best_r), xy=(0.2, 0.8),
                 xycoords='axes fraction', color='#101028',
             size=15)
plt.ylabel(y_name, size=18)
plt.xlabel("Model fit", size=18)
plt.show()
