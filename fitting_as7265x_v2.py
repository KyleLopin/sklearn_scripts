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
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


plt.style.use('seaborn')

data = pd.read_csv('mango_flouro_rows.csv')

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)


def regression_model(_y, _x1, _x2):
    ones = np.ones(len(_x1))
    #print(_x1)
    #print(_x2)
    x = np.column_stack((_x1, _x2, ones))
    # x = np.column_stack((_x1, ones))
    # print(x)
    x = sm.add_constant(x)
    model = sm.OLS(_y, x)
    results = model.fit()

    # print(results.summary())
    return results

# 'Total Chlorophyll (ug/ml)', 'Chlorophyll a (ug/ml)'
print(data.columns)
data_name = 'Total Chlorophyll (ug/ml)'

r2s = []
details = []
y = data[data_name]
y = 1 / y
best_r = 0
best_details = None
for i, data_column1 in enumerate(data_columns):
    for j, data_column2 in enumerate(data_columns):
        if j < i:
            continue
        x1 = data[[data_column1]]
        x2 = data[[data_column2]]
        # x1 = data[[data_column1, data_column2]]
        results = regression_model(y, x1, x2)
        r_sq_adj = results.rsquared_adj
        print('r^2 =', i, r_sq_adj, best_r, len(r2s), best_details)
        if r_sq_adj > 0.75:
            r2s.append(r_sq_adj)
            details.append((data_column1, data_column2))
        if r_sq_adj > best_r:
            best_r = r_sq_adj
            best_details = (data_column1, data_column2)

with open('as7265x_2_var_inv_ch_total.pickle', 'wb') as pkl:
    pickle.dump([r2s, details], pkl)

print(sorted(r2s))
