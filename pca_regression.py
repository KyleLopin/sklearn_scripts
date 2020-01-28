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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn')

data = pd.read_csv('mango_chloro_refl3.csv')

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)

y_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
             'Total Chlorophyll (ug/ml)']

x_data = data[data_columns]
print(x_data)
y_data = data[y_columns[0]]
y_data = 1 / y_data
# y_data = np.log(1 / y_data)
# y_data = np.exp(y_data)

pca = PCA(n_components=4)
X_standard = StandardScaler().fit_transform(x_data)
# X_r = pca.fit(x_data, y_data)
# print(pca.explained_variance_ratio_)
x_pca = pca.fit_transform(X_standard)

lin_model = linear_model.LinearRegression()
lin_model.fit(x_pca, y_data)

y_cal = lin_model.predict(x_pca)
y_cv = cross_val_predict(lin_model, x_pca, y_data, cv=10)

score_cal = r2_score(y_data, y_cal)
score_cv = r2_score(y_data, y_cv)

mse_cal = mean_squared_error(y_data, y_cal)
mse_cv = mean_squared_error(y_data, y_cv)

print(score_cal, score_cv, mse_cal, mse_cv)

plt.scatter(y_data, y_cal)
plt.show()
