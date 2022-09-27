# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.preprocessing import FunctionTransformer, StandardScaler
# local files
import get_data
plt.style.use("seaborn")


BACKGROUND = [687.65, 9453.7, 23218.35, 9845.05, 15496.7,
              18118.55, 7023.8, 7834.1, 28505.9, 4040.9,
              5182.3, 1282.55, 2098.85, 1176.1, 994.45,
              496.45, 377.55, 389.75]


def invert(x):
    return 1/x


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


x, y, fitting_data = get_data.get_data("mango", "as7265x", int_time=150,
                                       position=2, average=False, led="b'White IR'",
                                       led_current="25 mA", return_type="XYZ")
print(y)
y = y['Avg Total Chlorophyll (µg/cm2)']
print(y.shape)
regr = LinearRegression()
cv = RepeatedKFold(n_splits=5, n_repeats=10)
# regr = PLSRegression(n_components=12)
# regr = TransformedTargetRegressor(regressor=regr,
#                                   func=np.log,
#                                   inverse_func=np.exp)
# regr = TransformedTargetRegressor(regressor=regr,
#                                   func=neg_log,
#                                   inverse_func=neg_exp)

scores = cross_validate(regr, x, y, scoring=('r2', 'neg_mean_absolute_error'),
                        cv=cv, return_train_score=True)
print(scores.keys())
test_r2_mean = scores['test_r2'].mean()
test_r2_std = scores['test_r2'].std()

test_mae_mean = scores['test_neg_mean_absolute_error'].mean()
test_mae_std = scores['test_neg_mean_absolute_error'].std()

train_r2_mean = scores['train_r2'].mean()
train_r2_std = scores['train_r2'].std()

train_mae_mean = scores['train_neg_mean_absolute_error'].mean()
train_mae_std = scores['train_neg_mean_absolute_error'].std()

print("test r2: ", test_r2_mean, test_r2_std)
print("test mae: ", test_mae_mean, test_mae_std)

print("train r2: ", train_r2_mean, train_r2_std)
print("train mae: ", train_mae_mean, train_mae_std)

print("=====")
regr.fit(x, y)

y_predict = regr.predict(x)
print(y_predict.shape)

mae = mean_absolute_error(y, y_predict)
print(mae)
print(regr.score(x, y))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
regr.fit(X_train, y_train)

y_predict_train = regr.predict(X_train)
mae_train = mean_absolute_error(y_train, y_predict_train)
y_predict_test = regr.predict(X_test)
mae_test = mean_absolute_error(y_test, y_predict_test)

plt.scatter(y_train, y_predict_train, color='indigo', label="Training set")
plt.scatter(y_test, y_predict_test, color='forestgreen', label="Test set")
plt.title("AS7265X fits to mango leaves chlorophyll levels\nLinear Regression", size=18)
plt.plot([0, 90], [0, 90], c='mediumvioletred', ls='--')
plt.xlabel("Measured chlorophyll (µg/cm\u00B2)", size=15)
plt.ylabel("Predicted chlorophyll (µg/cm\u00B2)", size=15)
plt.annotate("R\u00B2 training set ={:.2f}".format(regr.score(X_train, y_train)), xy=(.05, 0.94),
                  xycoords='axes fraction', color='#101028', fontsize='x-large')
plt.annotate("Mean absolute error training set = {:.2f}".format(mae_train), xy=(.05, 0.87),
                  xycoords='axes fraction', color='#101028', fontsize='x-large')

plt.annotate("R\u00B2 training set ={:.2f}".format(regr.score(X_test, y_test)), xy=(.05, 0.80),
                  xycoords='axes fraction', color='#101028', fontsize='x-large')
plt.annotate("Mean absolute error training set = {:.2f}".format(mae_test), xy=(.05, 0.73),
                  xycoords='axes fraction', color='#101028', fontsize='x-large')
plt.legend(loc="lower right", prop={'size': 18})
plt.show()
