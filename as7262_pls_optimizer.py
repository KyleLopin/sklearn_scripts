# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
# local files
import processing

plt.style.use('seaborn')

data = pd.read_csv('as7262_roseapple.csv')
data = data.groupby('Leaf number', as_index=True).mean()

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)

y_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
             'Total Chlorophyll (ug/ml)']

x_data = data[data_columns]
# x_data = StandardScaler().fit_transform(x_data)
# x_data = np.exp(x_data)
# x_data = processing.snv(x_data)

print(x_data)
y_data = 1/data[y_columns[2]]

pls = PLSRegression(n_components=2)
pls.fit(x_data, y_data)

y_fit = pls.predict(x_data)
print(y_fit)
print(y_data.shape, y_fit.shape)
print(r2_score(y_data, y_fit))
print(mean_absolute_error(1/y_data, 1/y_fit))

plt.scatter(1/y_data, 1/y_fit)
plt.show()
