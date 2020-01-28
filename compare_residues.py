# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.metrics import mean_absolute_error, r2_score

plt.style.use('ggplot')

data = pd.read_csv('as7262_mango.csv')
res_data = pd.read_csv("mango_residues.csv")


data = data.loc[(data['position'] == 'pos 2')]
data = data.groupby('Leaf number', as_index=True).mean()
data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)

x_data = data[data_columns]

y_name = 'Total Chlorophyll (ug/ml)'
y_data = data[y_name]
res_data = res_data[y_name]
x_data_column = data["550 nm"]

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

fit_values, _ = curve_fit(model, x_data_column, y_data, maxfev=10 ** 6)

y_fit_model = model(x_data_column, *fit_values)

res_model = y_data - y_fit_model

# make residues from PLS
pls = PLS(n_components=3)
pls_model = pls.fit(x_data, y_data)
y_fit_pls = pls.predict(x_data)
print(y_fit_pls)
print(y_data)
print(type(y_data), type(y_fit_pls), type(y_fit_model))
print(y_data)
print(y_data.to_numpy())
print(y_fit_pls.T[0])
res_pls = y_data - y_fit_pls.T[0]

print(res_model)
print(type(res_model))
print(type(res_pls))
plt.hist(res_model, alpha=0.5, color="cadetblue", label="Single wavelenght model", bins=20)
plt.hist(res_pls, alpha=0.5, color="darkgoldenrod", label="PLS", bins=20)
print(res_data)
plt.hist(res_data, alpha=0.5, color="red", label="Data residues", weights=0.2*np.ones_like(res_data), bins=30)

plt.show()
print(res_model.shape, res_pls.shape)
print(res_pls)