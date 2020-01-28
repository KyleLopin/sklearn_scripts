# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler


def view_pls_coeff(x, y, num_pls_components, num_variables):
    pls = PLSRegression(num_pls_components)
    x_scaled_np = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_np, columns=x.columns)
    pls.fit(x_scaled, y)
    print(pls.coef_)
    print(type(pls.coef_))
    sorted_coeff = np.flip(np.argsort(np.abs(pls.coef_[:, 0])))
    print(sorted_coeff)
    print(np.flip(np.sort(np.abs(pls.coef_[:, 0]))))
    # ordered_coeffs = np.flip(np.sort(np.abs(pls.coef_[:, 0])))
    ordered_coeffs = np.abs(pls.coef_[:, 0])[sorted_coeff]
    print(x.columns)
    useful_columns = x.columns[sorted_coeff][:num_variables]
    print(useful_columns)
    plt.plot(ordered_coeffs)
    plt.show()


if __name__ == '__main__':
    data_full = pd.read_csv('wildbetal_flouro_rows.csv')
    # data_full = data_full[data_full['position'] == 'pos 2']
    y_name = 'Chlorophyll a (ug/ml)'
    y_data = data_full[y_name]

    x_data_columns = []
    for column in data_full:
        if 'nm' in column:
            x_data_columns.append(column)

    x_data = data_full[x_data_columns]

    view_pls_coeff(x_data, y_data, 40, 100)
