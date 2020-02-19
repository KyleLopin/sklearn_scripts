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
from sklearn.preprocessing import RobustScaler, StandardScaler
# local files
import data_getter
import processing


plt.style.use('seaborn')

# fitting_data = pd.read_csv('as7262_roseapple.csv')
fitting_data = pd.read_csv("as7262_mango.csv")
print(fitting_data.columns)
# fitting_data = fitting_data.loc[(fitting_data['integration time'] == 3)]
# fitting_data = fitting_data.loc[(fitting_data['position'] == 'pos 2')]
# print(fitting_data)
# fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()

# fitting_data = fitting_data.drop(["Leaf: 50"])

data_columns = []
for column in fitting_data.columns:
    if 'nm' in column:
        data_columns.append(column)
#
spectrum_data = fitting_data[data_columns]

spectrum_data, fitting_data = data_getter.get_data('mango', remove_outlier=True,
                                                   only_pos2=True)

# spectrum_data, _= processing.msc(spectrum_data)
# spectrum_data = processing.snv(spectrum_data)
# spectrum_data = pd.DataFrame(np.diff(spectrum_data))
spectrum_data = spectrum_data.diff(axis=1).iloc[:, 1:]
spectrum_data = pd.DataFrame(RobustScaler().fit_transform(spectrum_data))

spectrum_data.T.plot()
plt.show()
#
# print(spectrum_data["600 nm"])
# print(spectrum_data["600 nm"].idxmin())
# print(ham)
chloro_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']
y_name = 'Total Chlorophyll (ug/ml)'
predict_data = fitting_data[y_name]
print(spectrum_data)
# spectrum_data = spectrum_data.drop(["Leaf: 49", "Leaf: 35", "Leaf: 3"])
# # # spectrum_data = spectrum_data.drop([49, 35, 3])
# # print(predict_data)
# predict_data = predict_data.drop(["Leaf: 49", "Leaf: 35", "Leaf: 3"])
# # print(spectrum_data["600 nm"])
# print(spectrum_data["600 nm"].idxmin())
# print(ham)


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

    axis.scatter(_x, _y, s=20, c='mediumseagreen')
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
    axis.plot(x_linespace, y_fit_line, c='darkviolet')
    r2 = r2_score(y, y_fit)
    mae = mean_absolute_error(y, y_fit)
    print(wavelength, ',', r2, ',', mae)
    print(_x.idxmin(), _x.idxmax())

    y_top = y_fit_line + mae
    y_bottom = y_fit_line - mae
    axis.plot(x_linespace, y_top, 'k--', lw=1)
    axis.plot(x_linespace, y_bottom, 'k--', lw=1)
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


# y = predict_data
# x = spectrum_data[data_columns[3]]
# plt.scatter(x, y)
#
# fitting_model = poly_2_model
#
# fit_values, _ = curve_fit(fitting_model, x, y, maxfev=10000)
# print(fit_values)
# x_linespace = np.linspace(np.min(x), np.max(x))
# y_fit = fitting_model(x, *fit_values)
# y_linespace = fitting_model(x_linespace, *fit_values)
# print(x_linespace)
# # print(y_fit)
# plt.plot(x_linespace, y_linespace, c='r')
# r2 = r2_score(y, y_fit)
# mae = mean_absolute_error(y, y_fit)
# y_top = y_linespace + mae
# y_bottom = y_linespace - mae
# plt.plot(x_linespace, y_top, c='black')
# plt.plot(x_linespace, y_bottom, c='black')
# print(r2, mae)
# print(plt.xlim())
# _xy = (0.8*plt.xlim()[1], 0.8*plt.ylim()[1])
# print(_xy)
# plt.annotate(u'r\u00B2 = {0:.2f}\nMAE = {1:.4f}'.format(r2, mae),
#              xy=_xy, fontsize=12)
y = predict_data

models = [linear_model, log_model, exp_model, poly_2_model]

model_names = ['Linear model', "Logarithm model",
               "Exponential model", "Polynomial model"]
models = [linear_model, exp_model, poly_2_model]
model_names = ['Linear model',
               "Exponential model", "Polynomial model"]
# models = [linear_model, poly_2_model]
# model_names = ['Linear model', "Polynomial model"]

print(spectrum_data)
letters = ["A", "B", "C", "D", "E", "F"]

for i, model in enumerate(models):
    figure, axes, = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
    axes = [axes[0][0], axes[0][1], axes[1][0],
            axes[1][1], axes[2][0], axes[2][1]]
    figure.suptitle("{0}\n{1}".format(y_name, model_names[i]), size=20,
                    fontname='Franklin Gothic Medium')
    print("Model: ", model_names[i])

    for i in range(5):
        # x = spectrum_data[data_columns[i]]
        x = spectrum_data.iloc[:, i]
        letter = letters[i]
        # x = 1 / x
        put_xlabel = False
        if i >= 4:
            put_xlabel = True
        fit_n_plot(x, y, model, axes[i],
                   add_xlabel=put_xlabel,
                   figure_letter=letter,
                   wavelength=data_columns[i],
                   invert_y=False)


plt.show()
