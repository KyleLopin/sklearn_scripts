# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, PolynomialFeatures
# local files
import get_data

plt.style.use('seaborn')


# x_data, y_data = get_data.get_data_as726x_serial("mango", "as7262", int_times=[150],
#                                                  positions=[1, 2, 3],
#                                                  led_currents=["25 mA"])
x_data, y_data = get_data.get_data("mango", "as7262", int_time=[150],
                                   position=[1, 2, 3], led_current=["25 mA"])

print(x_data)
print('==')
print(y_data.to_string())
print('======')

y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
print(x_data.shape, y_data.shape)

pls = PLSRegression(n_components=6)


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2)
pls.fit(X_train, y_train)
y_test_predict = pls.predict(X_test)
y_train_predict = pls.predict(X_train)

plt.scatter(y_test_predict, y_test, color='forestgreen', label="Test set")
plt.scatter(y_train_predict, y_train, color='indigo', label="Training set")
y_line = [np.min(y_train_predict), np.max(y_train_predict)]
plt.plot(y_line, y_line, color='red', lw=1, linestyle='--')
plt.legend()

r2_test = r2_score(y_test, y_test_predict)
# r2_test=1
mae_test = mean_absolute_error(y_test, y_test_predict)
# mae_test = 1
r2_train = r2_score(y_train, y_train_predict)
# r2_train=1
mae_train = mean_absolute_error(y_train, y_train_predict)
# mae_train = 1
LEFT_ALIGN = 0.07
plt.annotate(u"R\u00B2 test ={:.2f}".format(r2_test), xy=(LEFT_ALIGN, 0.8),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.annotate("MAE test ={:.3f}".format(mae_test), xy=(LEFT_ALIGN, 0.68),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.annotate("MAE train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.62),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.title("Total Chlorophyll (µg/cm2)")
plt.ylabel("Measured Chlorophyll")
plt.show()

