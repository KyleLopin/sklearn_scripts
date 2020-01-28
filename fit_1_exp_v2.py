# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Functions to make data sets that are to be analyzed
"""

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
# local files
import helper_functions as funcs

plt.style.use('seaborn')

__author__ = "Kyle Vitatus Lopin"

# plt.xkcd()

refl_data = pd.read_csv('as7262_mango.csv')

# refl_data = refl_data.loc[(refl_data['position'] == 'pos 3')]
refl_data = refl_data.groupby('Leaf number', as_index=True).mean()

print(refl_data)
print('=========')
print(refl_data.columns)

data_columns = []
for column in refl_data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)
chloro_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']
y_name = 'Total Chlorophyll (ug/ml)'

def regression_model(_y, _x1):
    ones = np.ones(len(_x1))
    x = np.column_stack((_x1, ones))
    # print(x)
    x = sm.add_constant(x)
    model = sm.OLS(_y, x)
    results = model.fit()

    print(results.summary())
    return results


# for data_column in data_columns:
def fit_n_plot_data(data_column, axis, add_xlabel, figure_letter):
    # average_data.plot.scatter(data_column, y_name)
    x = refl_data[y_name]
    # y = np.log(1/y)
    # x = 1/x
    y = refl_data[data_column]
    # x = 1/x
    # y = 1 / y
    # x = np.log(1/x)
    print(x)
    results = regression_model(x, y)
    r_sq_adj = results.rsquared_adj
    print(results.params)
    fitted_x = (refl_data[data_column] * results.params.x1 +
                results.params.const)
    fitted_y = results.predict()
    # print(fitted_y)
    print(x.min(), x.max())
    # fit_x_space = np.linspace(x.min(), x.max())
    # print('x = ', fit_x_space)
    # fitted_y = (fit_x_space * results.params.x1 +
    #            results.params.const)

    axis.plot(x, y, 'o', color='mediumseagreen', markersize=4)
    # axis.plot(x, fitted_y, color='rebeccapurple')
    axis.annotate(u"R_adj\u00B2 =\n{:.3f}".format(r_sq_adj), xy=(0.1, 0.8),
                 xycoords='axes fraction', color='#101028')
    axis.annotate(figure_letter, xy=(-.2, .98),
                  xycoords='axes fraction', size=19,
                  weight='bold')
    wavelength = data_column.split('.')[0]
    # axis.set_xlabel("counts".format(wavelength))
    # axis.set_title("{0} nm measurement".format(wavelength))
    # axis.set_ylabel(y_name)
    axis.set_xlabel(y_name)
    print(wavelength)
    axis.title.set_text("{0} nm sensor channel".format(wavelength))
    # axis.set_ylim([0., 1.1])
    if add_xlabel:
        print('add y lable:', add_xlabel)
        # axis.set_xlabel('Sensor Counts')
        axis.set_xlabel(y_name)


figure, axes,  = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
print(axes)
figure.suptitle("{0} [ln(1/R)]".format(y_name.split('(')[0]), size=20,
                fontname='Franklin Gothic Medium')
axes = [axes[0][0], axes[0][1], axes[1][0],
        axes[1][1], axes[2][0], axes[2][1]]
letters = ['A', 'B', 'C', 'D', 'E', 'F']
for i, data_column in enumerate(data_columns):
    print(axes[i])
    print(data_column)
    print('fitting')
    fit_n_plot_data(data_column, axes[i], (i >= 4), letters[i])


# plt.tight_layout(0.5)
plt.show()



# sorted_data = average_data.sort_values(by=['Total Chlorophyll (ug/ml)'])
# # pd.set_option('display.max_columns', 500)
# print(sorted_data['Total Chlorophyll (ug/ml)'])
# print(sorted_data.index)

