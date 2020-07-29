# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import RobustScaler, StandardScaler

import data_getter

chloro_types = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                'Total Chlorophyll (ug/ml)']
x_data, y, data = data_getter.get_data('as7262 betal')
y = y[chloro_types]
y = np.exp(-y)
est = make_pipeline(RobustScaler(), PLSRegression(n_components=3))

est.fit(x_data, y)
Y_fit = est.predict(x_data)
Y_fit = Y_fit
print(Y_fit)
print(Y_fit.shape)
print(Y_fit[:, 0])
plt.scatter(Y_fit[:, 0], y.iloc[:, 0])
_line = np.linspace(y.iloc[:, 0].min(), y.iloc[:, 0].max())
plt.plot(_line, _line, 'k--')
plt.show()
