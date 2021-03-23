# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.decomposition import PCA
import get_data

plt.style.use('seaborn')

# data = pd.read_csv('mango_chloro_refl3.csv')
x_data, y_data = get_data.get_data("mango", "as7262", int_time=[150],
                                   position=[1, 2, 3], led_current=["25 mA"])
y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
# data_columns = []
# for column in data.columns:
#     if 'nm' in column:
#         data_columns.append(column)
# print(data_columns)
#
# y_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
#              'Total Chlorophyll (ug/ml)']
#
# x_data = data[data_columns]
# print(x_data)
# y_data = data[y_columns[0]]

pca = PCA(n_components=2)
X_r = pca.fit(x_data, y_data)
# print(pca.explained_variance_ratio_)
x_fit = pca.transform(x_data)
print(x_fit)
print(x_fit.shape)
print(x_fit[:, 0])

plt.scatter(x_fit[:, 0], x_fit[:, 1])

plt.show()
