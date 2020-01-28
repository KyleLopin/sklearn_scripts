# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


plt.style.use('seaborn')

fitting_data = pd.read_csv('as7262_roseapple.csv')
print(fitting_data.columns)
fitting_data = fitting_data.loc[(fitting_data['integration time'] == 5) | (fitting_data['integration time'] == 4)]
# fitting_data = fitting_data.loc[(fitting_data['integration time'] == 5)]
fitting_data = fitting_data.loc[(fitting_data['position'] == 'pos 2')]
fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()

pls = PLSRegression(n_components=6)
# pls = DecisionTreeRegressor(max_depth=5)

x_data_columns = []
for column in fitting_data:
    if 'nm' in column:
        x_data_columns.append(column)

y1 = fitting_data['Total Chlorophyll (ug/ml)']
y2 = fitting_data['Chlorophyll a (ug/ml)']
y3 = fitting_data['Chlorophyll b (ug/ml)']

x_data = fitting_data[x_data_columns]

x_data = np.log(x_data)

x_scaled_np = StandardScaler().fit_transform(x_data)

x_scaled = pd.DataFrame(x_scaled_np, columns=x_data.columns)

figure, axes, = plt.subplots(2, 2, figsize=(7.5, 8.75), constrained_layout=True)
axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

for i, y in enumerate([y1, y2, y3]):
    # y = 1 / y.values
    # print(1/y, 1/(1/y))
    pls.fit(x_scaled, y)
    y_predict = pls.predict(x_scaled)
    # y = 1 / y
    # y_predict = 1 / y_predict
    axes[i].scatter(y, y_predict)
    # axes[i].scatter(1/y, 1/(1/y))
    print(r2_score(y, y_predict), mean_absolute_error(y, y_predict))

plt.show()

def plt_
