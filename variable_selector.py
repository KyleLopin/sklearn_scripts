# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupShuffleSplit

data_full = pd.read_csv('mango_flouro_rows.csv')
data_full = data_full[data_full['position'] == 'pos 2']
y_name = 'Total Chlorophyll (ug/ml)'
y_data = data_full[y_name]

# print(data_full)
x_data_columns = []
for column in data_full:
    if 'nm' in column:
        x_data_columns.append(column)

x_data = data_full[x_data_columns].values
print(x_data)
print(type(x_data))
print(x_data.shape)
print(y_data.shape)


def pls_variable_selection(x, y, max_components, scorer, _cv, cv_groups):
    """

    Adopted from https://nirpyresearch.com/variable-selection-method-pls-python/
    :param x:
    :param y:
    :param max_components:
    :param scorer:
    :return:
    """

    # make a score table to fill in
    score_table = np.zeros( (max_components, x.shape[1]) )

    # try PLS with mulitple number of components
    for i in range(max_components):

        pls_i = PLSRegression( n_components=i+1 )
        pls_i.fit(x, y)

        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_indices = np.argsort( np.abs(pls_i.coef_[:, 0]) )
        print('=========')
        print(pls_i.coef_[:, 0])
        print(sorted_indices)
        print(type(sorted_indices))
        print(x)
        # sort the columns of the x data
        x_sorted = x[:, sorted_indices]

        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        for j in range(x_sorted.shape[-1]-(i+1)):

            pls_j = PLSRegression( n_components=i+1 )
            pls_j.fit(x_sorted[:, j:], y)

            y_cv = cross_val_predict(pls_j, x_sorted[:, j:], y, cv=3)

            score_table[i, j] = r2_score(y, y_cv)

            print('i, j, score = ', i, j, score_table[i, j])


    score_table.tofile('pls_score_table2.dat')
    mseminx, mseminy = np.where(score_table == np.max(score_table[np.nonzero(score_table)]))
    print("Optimised number of PLS components: ", mseminx[0] + 1)
    print("Wavelengths to be discarded ", mseminy[0])
    print('Optimised MSEP ', score_table[mseminx, mseminy][0])


if __name__ == '__main__':
    y = np.array([1, 2, 3])
    x = np.array([[1, 2, 3], [10, 20, 30], [15, 27, 38]])
    cv_splitter = GroupShuffleSplit(n_splits=100,
                                    test_size=0.35,
                                    random_state=120)
    group_splitter = data_full['Leaf number']
    pls_variable_selection(x_data, y_data, 25, r2_score, cv_splitter, group_splitter)
