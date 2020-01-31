# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# local files
import data_getter
import processing

plt.style.use('ggplot')

x_data, data = data_getter.get_data('as7263 mango')
print(data.columns)
print('(((((((')
print(data['integration time'].unique(), data['position'].unique())
x_data = x_data[data['integration time'] == 200]
x_data = x_data[data['position'] == 'pos 2']
data = data[data['integration time'] == 200]
data = data[data['position'] == 'pos 2']
chloro_data = data.groupby('Leaf number', as_index=True).mean()

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)
print(x_data)
print(data)
print(chloro_data)

y_columns = ['Total Chlorophyll (ug/ml)',
             'Chlorophyll a (ug/ml)',
             'Chlorophyll b (ug/ml)']

def linear_model(x, a, b):
    return a * x + b


def log_model(x, a, b, c):
    if b < 0:
        return -1000000
    return a * np.log(b * x) + c


def exp_model(x, a, b, c):
    return a * np.exp(-b * x) + c


def poly_2_model(x, a, b, c):
    return a * x**2 + b * x + c


def fit_n_plot(_x, _y, model, axis, add_xlabel=None,
               figure_letter=None, wavelength=None,
               invert_y=False):

    axis.scatter(_x, _y)
    if invert_y:
        _y = 1 / _y
    fit_values, _ = curve_fit(model, _x, _y, maxfev=10**6)
    x_linespace = np.linspace(np.min(_x), np.max(_x))

    y_fit = model(_x, *fit_values)
    y_fit_line = model(x_linespace, *fit_values)
    axis.plot(x_linespace, y_fit_line, c='r')
    # print(y_fit)
    r2 = r2_score(y, y_fit)
    mae = mean_absolute_error(y, y_fit)
    print(wavelength, ',', r2, ',', mae)
    print(_x.idxmin(), _x.idxmax())

    y_top = y_fit_line + mae
    y_bottom = y_fit_line - mae
    axis.plot(x_linespace, y_top, c='black')
    axis.plot(x_linespace, y_bottom, c='black')
    axis.annotate(u"R\u00B2 ={:.3f}".format(r2), xy=(0.7, 0.85),
                  xycoords='axes fraction', color='#101028')
    axis.annotate(u"MAE ={:.3f}".format(mae), xy=(0.7, 0.75),
                  xycoords='axes fraction', color='#101028')
    axis.annotate(figure_letter, xy=(-0.2, 0.98),
                  xycoords='axes fraction', size=19,
                  weight='bold')
    axis.set_ylabel(y_name)
    if add_xlabel:
        axis.set_xlabel("Fraction Reflectance")
    axis.title.set_text("{0} sensor channel".format(wavelength))



models = [linear_model, log_model, exp_model, poly_2_model]

model_names = ['Linear model', "Logarithm model",
               "Exponential model", "Polynomial model"]
# models = [linear_model]
# model_names = ['Linear model']

chloro_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']
y_name = ['Total Chlorophyll (ug/ml)']

letters = ["A", "B", "C", "D", "E", "F"]
print(x_data.index)
x_data = processing.snv(x_data)

best_score = np.inf
best_conditions = None
good_sets = []

invert_y = True
# for led in data['LED'].unique():
#     print(led)
#     for i, y_name in enumerate(chloro_columns):
#         for j, model in enumerate(models):
#
#             # figure, axes, = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
#             # axes = [axes[0][0], axes[0][1], axes[1][0],
#             #         axes[1][1], axes[2][0], axes[2][1]]
#             # figure.suptitle("{0} measured with AS7263\n and {1}".format(y_name, led), size=20,
#             #                 fontname='Franklin Gothic Medium')
#
#             for l in range(6):
#                 x = x_data[data_columns[l]]
#                 letter = letters[l]
#                 # x = 1 / x
#                 put_xlabel = False
#                 if l >= 4:
#                     put_xlabel = True
#                 y = chloro_data[y_name]
#                 x = x_data[data['LED'] == led]
#                 x = x[data_columns[l]]
#
#                 try:
#
#                     # fit_n_plot(x, y, model, axes[l],
#                     #            add_xlabel=put_xlabel,
#                     #            figure_letter=letter,
#                     #            wavelength=data_columns[l],
#                     #            invert_y=False)
#                     if invert_y:
#                         y = 1 / y
#                     fit_values, _ = curve_fit(model, x, y, maxfev=10 ** 6)
#                     y_fit = model(x, *fit_values)
#                     if invert_y:
#                         y = 1 / y
#                         y_fit = 1 / y_fit
#                     r2 = r2_score(y, y_fit)
#                     mae = mean_absolute_error(y, y_fit)
#                     print(r2, mae)
#                 except:
#                     pass

x_msc, _ = processing.msc(x_data)
x_inv_msc = 1 / x_msc.copy()
data_sets = [x_data.copy(), 1 / x_data, processing.snv(x_data), 1 / processing.snv(x_data), x_msc, x_inv_msc,
             StandardScaler().fit_transform(x_data), StandardScaler().fit_transform(processing.snv(x_data)),
             StandardScaler().fit_transform(x_msc), RobustScaler().fit_transform(x_data),
             RobustScaler().fit_transform(processing.snv(x_data)),
             RobustScaler().fit_transform(x_msc)]
data_set_names = ["raw", "inverse", "SNV", "Invert SNV", "MSC", "inverse msc",
                  "standard scalar", "Standard Scalar SNV", "Standard Scalar MSC",
                  "Robust Scalar", "Robust Scalar SNV", "Robust Scalar MSC"]

for z, x_data in enumerate(data_sets):
    for led in data['LED'].unique():
        print(led)
        for y_invert in [True, False]:
            for j, model in enumerate(models):

                # figure, axes, = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
                # axes = [axes[0][0], axes[0][1], axes[1][0],
                #         axes[1][1], axes[2][0], axes[2][1]]
                # figure.suptitle("{0} measured with AS7263\n and {1}".format(y_name, led), size=20,
                #                 fontname='Franklin Gothic Medium')
                print(len(data_columns))
                for l in range(len(data_columns)):

                    try:

                        y = chloro_data[y_name].iloc[:, 0]
                        x = x_data[data['LED'] == led]
                        x = x[data_columns[l]]

                        if invert_y:
                            y = 1 / y

                        fit_values, _ = curve_fit(model, x, y, maxfev=10 ** 6)
                        y_fit = model(x, *fit_values)
                        if invert_y:
                            y = 1 / y
                            y_fit = 1 / y_fit
                        r2 = r2_score(y, y_fit)
                        mae = mean_absolute_error(y, y_fit)
                        print(led, model_names[j], y_invert,
                              r2, mae, data_set_names[z])
                        if mae < best_score:
                            good_sets.append((best_score, best_conditions))

                            best_score = mae
                            best_conditions = [led, y_invert, model_names[j],
                                               data_columns[l], data_set_names[z]]
                        print(best_score, best_conditions)
                    except Exception as error:
                        print(error)

print(good_sets)