# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Analysis of the intial flourescense measurements of mango leaves
using the as7265x and custom LED stimulating board.
"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


_data = pd.read_csv('flouro_mango_leaves_pos2.csv')

print(_data)
print(_data.columns)
data_columns = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm', '560 nm',
                '585 nm', '610 nm', '645 nm', '680 nm', '705 nm', '730 nm', '760 nm',
                '810 nm', '860 nm', '900 nm', '940 nm']
wavelengths = [int(i.split(' ')[0]) for i in data_columns]
print(wavelengths)
WAVELENGTHS1 = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm']
WAVELENGTHS2 = ['560 nm', '585 nm', '610 nm', '645 nm', '680 nm', '705 nm']
WAVELENGTHS3 = ['730 nm', '760 nm', '810 nm', '860 nm', '900 nm', '940 nm']


# 'Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
#        'Total Chlorophyll (ug/ml)'
y_data = 'Total Chlorophyll (ug/ml)'
chloro_columns = ['Chlorophyll a (ug/ml)',     'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']
y_name = 'Total Chlorophyll (ug/ml)'


LEDs = ['White LED', 'IR LED', 'UV (405 nm) LED', '390 nm LED', '395 nm LED',
        '400 nm LED', '405 nm LED', '410 nm LED', '425 nm LED', '455 nm LED',
        '465 nm LED', '470 nm LED', '475 nm LED', '480 nm LED', '505 nm LED',
        '525 nm LED', '630 nm LED', '890 nm LED', '940 nm LED']

int_times = [150, 200, 250]


def reg_m(_y, _x):
    ones = np.ones(len(_x))
    _X = sm.add_constant(np.column_stack((_x, ones)))
    _results = sm.OLS(_y, _X).fit()
    return _results


r_squared_fits = dict()


def fit(data_frame, _data_column_name):
    y = data_frame[y_data]
    y = np.log(1/y)
    print('y = ', y)
    # y = y / _background
    # print('y = ', y )
    x = data_frame[_data_column_name]
    results = reg_m(y, x)
    return results.rsquared_adj


def fit_all_data():
    best_r_squared = 0
    best_r_squared_details = None
    for int_time in int_times:
        r_squared_fits[str(int_time)] = dict()
        int_time_data_frame = _data.loc[_data['integration time'] == int_time]
        print(int_time_data_frame)
        for led in LEDs:
            print('led = ', led)
            led_dataframe = int_time_data_frame.loc[int_time_data_frame['LED'] == led]
            print(led_dataframe)
            r_squared_fits[str(int_time)][led] = []
            for column in data_columns:
                r_sq = fit(led_dataframe, column)
                r_squared_fits[str(int_time)][led].append(r_sq)
                if r_sq > best_r_squared:
                    best_r_squared = r_sq
                    best_r_squared_details = (int_time, led, column)
    return best_r_squared, best_r_squared_details


# pd.set_option('display.max_columns', 500)
# r_sq, r_sq_det = fit_all_data()
# print(r_squared_fits)
# print(r_sq)
# print(r_sq_det)
# print(r_squared_fits['150'])
# print(r_squared_fits['200'])
# print(r_squared_fits['250'])
#
# for led in LEDs:
#     print(led)
#     max_r_sq = max(r_squared_fits['250'][led])
#     max_index = [i for i, j in enumerate(r_squared_fits['250'][led]) if j == max_r_sq]
#     print(max_r_sq, data_columns[max_index[0]])

led = "505 nm LED"
# good = ["White LED"]
# led = LEDs[8]
int_time_data_frame = _data.loc[_data['integration time'] == 250]
data_frame = int_time_data_frame.loc[int_time_data_frame['LED'] == led]


def fit_n_plot(data_column, axis,add_xlabel, figure_letter):
    y = data_frame[y_data]
    y = np.log(1 / y)
    print('y = ', y)
    # y = y / _background
    # print('y = ', y )
    x = data_frame[data_column]
    results = reg_m(y, x)
    fitted_y = results.predict()
    axis.plot(x, y, 'o', color='mediumseagreen', markersize=4)
    axis.plot(x, fitted_y, color='rebeccapurple')
    r_sq_adj = results.rsquared_adj
    axis.annotate(u"R_adj\u00B2 =\n{:.3f}".format(r_sq_adj), xy=(0.7, 0.8),
                  xycoords='axes fraction', color='#101028')
    axis.annotate(figure_letter, xy=(-.2, .98),
                  xycoords='axes fraction', size=19,
                  weight='bold')
    wavelength = data_column.split('.')[0]
    # axis.set_xlabel("counts".format(wavelength))
    # axis.set_title("{0} nm measurement".format(wavelength))
    axis.set_ylabel(y_name)
    print(wavelength)
    axis.title.set_text("{0} nm sensor channel".format(wavelength))
    # axis.set_ylim([0., 1.1])
    if add_xlabel:
        print('add y label:', add_xlabel)
        axis.set_xlabel('Reflectance')

figure1, axes1 = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)

figure2, axes2 = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
figure3, axes3 = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)

letters = ['A', 'B', 'C', 'D', 'E', 'F']
def process_wavelenghts(wavelengths, _axes):
    axes = [_axes[0][0], _axes[0][1], _axes[1][0],
            _axes[1][1], _axes[2][0], _axes[2][1]]

    for i, wavelength in enumerate(wavelengths):
        axis = axes[i]
        fit_n_plot(wavelength, axis, False, letters[i])

process_wavelenghts(WAVELENGTHS1, axes1)
process_wavelenghts(WAVELENGTHS2, axes2)
process_wavelenghts(WAVELENGTHS3, axes3)
print(led)
plt.show()

# figure.suptitle("{0} ".format(y_name.split('(')[0]), size=28,
#                fontname='Franklin Gothic Medium')


# for i, data_column in enumerate(data_columns):
#     print(axes[i])
#     print(data_column)
#     print('fitting')
#     fit_n_plot_data(data_column, axes[i], (i >= 4), letters[i], background[i])


