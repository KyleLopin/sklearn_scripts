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

plt.style.use('seaborn')

data = pd.read_csv('as7262_mango.csv')

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)

y_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
             'Total Chlorophyll (ug/ml)']

x_data = data[data_columns]
print(x_data)
y_data = data[y_columns[2]]

pls = PLSRegression(n_components=2)
pls.fit(x_data, y_data)

y_fit = pls.transform(x_data)

print(r2_score(y_data, y_fit))
print(mean_absolute_error(y_data, y_fit))

plt.scatter(y_data, y_fit)
plt.show()
