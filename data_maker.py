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

as7262_files = funcs.get_all_files_with_stub('as7265x', 'betal')
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
#total_data

print(chlor_data.columns)
for _data in chlor_data.columns:
    average_data[_data] = chlor_data[_data]
# pd.set_option('display.max_columns', 500)
print('===========')
print(average_data)
print(average_data.columns)

data_columns = ['450.1', '500.1', '550.1', '570.1', '600.1', '650.1']
chloro_columns = ['Chlorophyll a (ug/ml)',     'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']
y_name = 'Total Chlorophyll (ug/ml)'

data = pd.DataFrame()


def reg_m(_y, _x):
    ones = np.ones(len(_x))
    _X = sm.add_constant(np.column_stack((_x, ones)))
    _results = sm.OLS(_y, _X).fit()
    return _results


# for data_column in data_columns:
def fit_n_plot_data(data_column, axis, add_xlabel, figure_letter,
                    _background):
    # average_data.plot.scatter(data_column, y_name)
    y = average_data[y_name]
    print('y = ', y)
    # y = y / _background
    # print('y = ', y )
    x = average_data[data_column]
    print('x = ', x)
    x = x / _background
    print('x = ', x)

    results = reg_m(y, x)
    r_sq_adj = results.rsquared_adj
    print(results.params)
    fitted_x = (average_data[data_column] * results.params.x1 +
                results.params.const)
    fitted_y = results.predict()

    axis.plot(x, y, 'o', color='mediumseagreen', markersize=4)
    axis.plot(x, fitted_y, color='rebeccapurple')
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
    axis.set_ylim([0., 1.1])
    if add_xlabel:
        print('add y lable:', add_xlabel)
        axis.set_xlabel('Reflectance')


figure, axes,  = plt.subplots(3, 2, figsize=(7.5, 8.75), constrained_layout=True)
print(axes)
figure.suptitle("{0} ".format(y_name.split('(')[0]), size=28,
                fontname='Franklin Gothic Medium')
axes = [axes[0][0], axes[0][1], axes[1][0],
        axes[1][1], axes[2][0], axes[2][1]]
letters = ['A', 'B', 'C', 'D', 'E', 'F']
for i, data_column in enumerate(data_columns):
    print(axes[i])
    print(data_column)
    print('fitting')
    fit_n_plot_data(data_column, axes[i], (i >= 4), letters[i], background[i])


# plt.tight_layout(0.5)
plt.show()



# sorted_data = average_data.sort_values(by=['Total Chlorophyll (ug/ml)'])
# # pd.set_option('display.max_columns', 500)
# print(sorted_data['Total Chlorophyll (ug/ml)'])
# print(sorted_data.index)

