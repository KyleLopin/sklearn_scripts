# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Script to import data (into refl_data), fit the data to
a linear regression model and plot the data for 6 different
spectrum and the x and y error bars from the chlorophyll
data collected in the summer of 2019 from an AS7262
"""

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# local files
import helper_functions as funcs

plt.style.use('seaborn')

__author__ = "Kyle Vitatus Lopin"

# refl_data = pd.read_csv('mango_chloro_refl3.csv')
# refl_data = pd.read_csv('betal_as7265x_full.csv')
# refl_data = pd.read_csv("wildbetal_flouro_rows.csv")
# refl_data = refl_data.loc[(refl_data['LED current'] == 50)]
print(refl_data)
refl_data = refl_data.loc[(refl_data['position'] == ' pos 2')]
print(refl_data['LED'].iloc[0])
print(refl_data['LED'].iloc[0]==' 505 nm')
print(type(refl_data['LED'].iloc[0]))
print((refl_data['LED'] == ' White LED'))
refl_data = refl_data.loc[(refl_data['LED'] == ' 425 nm LED')]
refl_data = refl_data.loc[(refl_data[' integration time'] == 200)]
# refl_data = refl_data.groupby('Leaf number', as_index=True).mean()

print(refl_data)
print('=========')
print(refl_data.columns)

data_columns = []
for column in refl_data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)
# data_columns = ['610 nm', '680 nm', '730 nm', '760 nm', '810 nm', '860 nm', '560 nm', '585 nm', '645 nm', '705 nm', '900 nm', '940 nm', '410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm']
data_columns = ['610 nm', '560 nm', '585 nm', '645 nm', '680 nm', '535 nm']
# data_columns = ['560 nm', '585 nm', '645 nm', '485 nm', '510 nm', '535 nm']
def regression_model(_y, _x1):
    ones = np.ones(len(_x1))
    x = np.column_stack((_x1, ones))
    print(x)
    x = sm.add_constant(x)
    model = sm.OLS(_y, x)
    results = model.fit()

    print(results.summary())
    return results

# regression_model(refl_data['Total Chlorophyll (ug/ml)'],
#                  refl_data['550 nm'])
# 'Total Chlorophyll (ug/ml)', 'Chlorophyll a (ug/ml)'
data_name = 'Total Chlorophyll (ug/ml)'

def fit_n_plot_data(data_column, axis, add_xlabel,
                    figure_letter):
    y = refl_data[data_name]
    x = refl_data[data_column]
    # =================================
    # Transform data here
    # ===================
    # y = np.log(y)

    # y = 1 / y
    # x = 1 / x
    print('y = ', y)
    print('x = ', x)
    results = regression_model(y, x)
    r_sq_adj = results.rsquared_adj
    print(results.params)
    prstd, iv_l, iv_u = wls_prediction_std(results)
    print(prstd)
    error_name = data_name+' STDEV'
    yerror = refl_data[error_name]
    axis.scatter(x, y, marker='o', color='mediumseagreen', s=20)

    # axis.errorbar(x, y, yerr=yerror, fmt='o', color='mediumseagreen', markersize=4)
    axis.plot(x, results.fittedvalues, color='rebeccapurple')


    # axis.plot(x, iv_u, 'red', alpha=0.2, dash_capstyle='round')
    # axis.plot(x, iv_l, 'red', alpha=0.2,
    #           linestyle=(0, (0.1, 2)), dash_capstyle='round')

    axis.annotate(u"R_adj\u00B2 =\n{:.3f}".format(r_sq_adj), xy=(0.7, 0.8),
                  xycoords='axes fraction', color='#101028')
    axis.annotate(figure_letter, xy=(-.2, .98),
                  xycoords='axes fraction', size=19,
                  weight='bold')
    wavelength = data_column.split('.')[0]
    axis.set_ylabel(data_name)
    print(wavelength)
    axis.title.set_text("{0} nm sensor channel".format(wavelength))
    # axis.set_ylim([0., 1.1])
    if add_xlabel:
        print('add y lable:', add_xlabel)
        axis.set_xlabel('Fraction of Reflectance')


figure, axes,  = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)

figure.suptitle("{0} ".format(data_name.split('(')[0]), size=28,
                fontname='Franklin Gothic Medium')
axes = [axes[0][0], axes[0][1], axes[1][0],
        axes[1][1], axes[2][0], axes[2][1]]
letters = ['A', 'B', 'C', 'D', 'E', 'F']

print(refl_data)
for i, data_column in enumerate(data_columns):
    print(data_column)
    print('fitting')
    fit_n_plot_data(data_column, axes[i], (i >= 4), letters[i])


plt.show()
