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

# local files
import data_getter
import processing

plt.style.use('ggplot')

x_data, data = data_getter.get_data('as7263 mango')
chloro_data = data.groupby('Leaf number', as_index=True).mean()

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)

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
    if invert_y:
        y_fit = 1/model(_x, *fit_values)
        y_fit_line = 1/model(x_linespace, *fit_values)
    else:
        y_fit = model(_x, *fit_values)
        y_fit_line = model(x_linespace, *fit_values)
    axis.plot(x_linespace, y_fit_line, c='r')
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
# models = [linear_model, exp_model, poly_2_model]
# model_names = ['Linear model',
#                "Exponential model", "Polynomial model"]

for model in models:
    for i, led in enumerate(data['LED'].unique()):
        figure, axes, = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
        axes = [axes[0][0], axes[0][1], axes[1][0],
                axes[1][1], axes[2][0], axes[2][1]]
        figure.suptitle("{0}\n{1}".format(y_name, model_names[i]), size=20,
                        fontname='Franklin Gothic Medium')
