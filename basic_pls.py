# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn')

data = pd.read_csv('as7262_roseapple.csv')
data = data.loc[(data['current'] == 50)]
data = data.loc[(data['position'] == 'pos 2')]
data = data.groupby('Leaf number', as_index=True).mean()

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
# y_data = 1 / y_data
# y_data = np.log(1 / y_data)
# y_data = np.exp(y_data)

pls = PLS(n_components=3)

model_fit = pls.fit(x_data, y_data)
y_predict = pls.predict(x_data)
# r2_test = r2_score(1/y_data, 1/y_predict)
# mae = mean_absolute_error(1/y_data, 1/y_predict)
# plt.scatter(1/y_data, 1/y_predict)

plt.scatter(y_data, y_predict)
r2_test = r2_score(y_data, y_predict)
mae = mean_absolute_error(y_data, y_predict)
print(r2_test, mae)
plt.xlim(0, 1)
plt.ylim(0, 1)
x_line = np.linspace(0.1, 1.0, 100)
plt.plot(x_line, x_line, 'orangered')
plt.plot(x_line, x_line+mae, 'r--')
plt.plot(x_line, x_line-mae, 'r--')

plt.show()





