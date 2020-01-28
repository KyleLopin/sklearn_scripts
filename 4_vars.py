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

plt.style.use('ggplot')

__author__ = "Kyle Vitatus Lopin"


def make_file_list():
    """ Make a small GUI for user to select all the data files to be analzed and return the list
    of filenames """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    filenames = filedialog.askopenfilename(initialdir="/", title="Select files to analze",
                                           filetypes=(("xcel files", "*.xlsx"), ("all files", "*.*")))
    print(filenames)


def get_all_files_with_stub(stub):
    """ Find all files in current directory that have the stub in their name """
    named_files = []
    for root, directory, files in os.walk(os.getcwd()):
        for filename in files:
            if stub in filename:
                print(root, filename)
                named_files.append(root + '/' + filename)
    return named_files


as7262_files = get_all_files_with_stub('as7262')
print(as7262_files)

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

chloro_data_filename = get_all_files_with_stub('absorbance')[0]
chlor_data = pd.read_excel(chloro_data_filename, sheet_name='Summary')
print(chlor_data)
# print(chlor_data.index)
# print(chlor_data)
chlor_data['leaf number:'] = 'Leaf: ' + chlor_data['leaf number:'].astype(str)
print(chlor_data.columns)
# chlor_data.rename(columns={'leaf number:', 'Leaf number'}, inplace=True)
# chlor_data.rename(columns={'Leaf number', 'leaf number:'}, inplace=True)
chlor_data['Leaf number'] = chlor_data['leaf number:']
del chlor_data['leaf number:']
chlor_data.set_index('Leaf number', inplace=True)
print(chlor_data)

# average_data.insert(len(average_data.columns),)
# total_data = average_data.merge(chlor_data, left_index=True, right_index=True)
# total_data

print(chlor_data.columns)
for _data in chlor_data.columns:
    average_data[_data] = chlor_data[_data]
# pd.set_option('display.max_columns', 500)
print('===========')
print(average_data)
print(average_data.columns)

data_columns = ['450.1', '500.1', '550.1', '570.1', '600.1', '650.1']
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

for i, column_i in enumerate(data_columns):
    for j, column_j in enumerate(data_columns):
        for k, column_k in enumerate(data_columns):
            # for m, column_m in enumerate(data_columns):
            # average_data.plot.scatter(data_column, 'Total Chlorophyll (ug/ml)')
            y = average_data['Total Chlorophyll (ug/ml)']
            # x = average_data[data_column]
            # lin_fit = LinearRegression.fit(average_data.loc[:, data_column],
            #                                average_data.loc[:, 'Total Chlorophyll (ug/ml)'])
            # print(LinearRegression().score(average_data[data_column, :],
            #                                average_data['Total Chlorophyll (ug/ml)', :]))
            results = reg_m(y, average_data[[column_i, column_j,
                                             column_k]])
            r_sq_adj = results.rsquared_adj
            print(r_sq_adj)

            if r_sq_adj > best_r:
                best_r = r_sq_adj
                best_params = results.params
                best_columns = [column_i, column_j, column_k]
            # fitted_x = (average_data[data_column] * results.params.x1 +
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
