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

from sklearn.model_selection import cross_validate, train_test_split, GroupKFold
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
x_data, y_data = get_data.get_data("mango", "as7262", int_time=[100, 200],
                                   position=[2, 3], led_current=["25 mA"])
x_data, y_data = get_data.get_data("mango", "as7265x", int_time=[100, 200], led="White",
                                   position=[2, 3], led_current=["25 mA"])

print(x_data)
print('==')
print(y_data.to_string())
print('======')

y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
print(x_data.shape, y_data.shape)
# x_data = StandardScaler().fit_transform(x_data)
x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)
# x_data = StandardScaler().fit_transform(x_data)
x_data = pd.DataFrame(x_scaled_np, index=x_data.index)


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


pls = PLSRegression(n_components=6)
pls = TransformedTargetRegressor(regressor=pls,
                                 func=neg_log,
                                 inverse_func=neg_exp)

cv = GroupKFold(n_splits=5)
# X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,
#                                                     test_size=0.2)
train_index, test_index, _, _ = train_test_split(np.arange(1, 101), np.arange(1, 101),
                                                    test_size=0.2)
print('l', train_index)
print('p', test_index)
print(x_data)
print(len(train_index), len(test_index))
X_train = x_data.loc[train_index]
X_test = x_data.loc[test_index]

y_train = y_data.loc[train_index]
y_test = y_data.loc[test_index]
print('+++')
print(y_train)



pls.fit(X_train, y_train)
y_test_predict = pls.predict(X_test)
y_train_predict = pls.predict(X_train)
y_test_predict = pd.DataFrame(y_test_predict, index=y_test.index)
y_train_predict = pd.DataFrame(y_train_predict, index=y_train.index)

print(y_test_predict)

y_test_predict = y_test_predict.groupby("Leaf number").mean()
y_train_predict = y_train_predict.groupby("Leaf number").mean()
y_test_mean = y_test.groupby("Leaf number").mean()
y_train_mean = y_train.groupby("Leaf number").mean()
print(y_test_predict)

plt.scatter(y_test_predict, y_test_mean, color='forestgreen', label="Test set")
plt.scatter(y_train_predict, y_train_mean, color='indigo', label="Training set")
y_line = [np.min(y_train_predict), np.max(y_train_predict)]
plt.plot(y_line, y_line, color='red', lw=1, linestyle='--')
plt.legend()

r2_test = r2_score(y_test_mean, y_test_predict)
# r2_test=1
mae_test = mean_absolute_error(y_test_mean, y_test_predict)
# mae_test = 1
r2_train = r2_score(y_train_mean, y_train_predict)
# r2_train=1
mae_train = mean_absolute_error(y_train_mean, y_train_predict)
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

