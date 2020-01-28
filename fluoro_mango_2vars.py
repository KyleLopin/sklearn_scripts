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
y_name = 'Chlorophyll b (ug/ml)'
y_data = y_name

LEDs = ['White LED', 'IR LED', 'UV (405 nm) LED', '390 nm LED', '395 nm LED',
        '400 nm LED', '405 nm LED', '410 nm LED', '425 nm LED', '455 nm LED',
        '465 nm LED', '470 nm LED', '475 nm LED', '480 nm LED', '505 nm LED',
        '525 nm LED', '630 nm LED', '890 nm LED', '940 nm LED']

# int_times = [150, 200, 250]
int_times = [250]

def reg_m(_y, _x):
    ones = np.ones(len(_x))
    _X = sm.add_constant(np.column_stack((_x, ones)))
    _results = sm.OLS(_y, _X).fit()
    return _results


r_squared_fits = dict()


def fit(data_frame, _data_column_name1, _data_column_name2):
    y = data_frame[y_data]
    y = np.log(1/y)
    # y = y / _background
    # print('y = ', y )
    x = data_frame[[_data_column_name1, _data_column_name2]]
    results = reg_m(y, x)
    return results.rsquared


def fit_all_data():
    best_r_squared = 0
    best_r_squared_details = None
    for int_time in int_times:
        r_squared_fits[str(int_time)] = dict()
        int_time_data_frame = _data.loc[_data['integration time'] == int_time]
        # print(int_time_data_frame)
        for led in LEDs:
            # print('led = ', led)
            led_dataframe = int_time_data_frame.loc[int_time_data_frame['LED'] == led]
            # print(led_dataframe)
            r_squared_fits[str(int_time)][led] = []
            for i, column1 in enumerate(data_columns):
                for j, column2 in enumerate(data_columns):
                    if j > i:
                        continue
                # for column3 in data_columns:
                    r_sq = fit(led_dataframe, column1, column2)
                    print(column1, column2, led, r_sq, best_r_squared, best_r_squared_details)

                    r_squared_fits[str(int_time)][led].append(r_sq)
                    if r_sq > best_r_squared:
                        best_r_squared = r_sq
                        best_r_squared_details = (int_time, led, column1, column2)
    return best_r_squared, best_r_squared_details

led_combo = []
for i, column1 in enumerate(data_columns):
    for j, column2 in enumerate(data_columns):
        if j > i:
            continue
        led_combo.append('{0} and {1}'.format(column1, column2))

# pd.set_option('display.max_columns', 500)
r_sq, r_sq_det = fit_all_data()
print(r_squared_fits)
print(r_sq)
print(r_sq_det)
for int_time in int_times:
    print(r_squared_fits[str(int_time)])

for led in LEDs:
    print(led)
    for int_time in int_times:

        max_r_sq = max(r_squared_fits[str(int_time)][led])
        max_index = [i for i, j in enumerate(r_squared_fits[str(int_time)][led]) if j == max_r_sq]
        print(max_index)
        print(max_r_sq, led_combo[max_index[0]], int_time)

