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
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLassoCV
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                   FunctionTransformer, PolynomialFeatures, KBinsDiscretizer)
from sklearn.svm import SVR
# local files
import get_data

plt.style.use('ggplot')

fruit = "mangoes"
fruit = "mango"
y_name = "Avg Total Chlorophyll (µg/cm2)"

hplc_fruits = ["mangoes", "tomato"]
index_name = "Leaf number"
average = True
if fruit in hplc_fruits:
    index_name = "Fruit"
    # y_name = 'lycopene (DW)'
    # y_name = "beta-carotene (DW)"
    y_name = "%DM"
    y_name = "phenols (FW)"


cv = GroupKFold(n_splits=5)
# lycopene
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=6)
train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
                                                 test_size=0.2, random_state=13)
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=5)
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=8)


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


pls = PLSRegression(n_components=6)
pls = TransformedTargetRegressor(regressor=pls,
                                 func=neg_log,
                                 inverse_func=neg_exp)


def make_1figure(pls, sensor, axis, led=None):
    if sensor == "c12880":
        x_data, y_data = get_data.get_data(fruit, sensor)
    else:
        x_data, y_data = get_data.get_data(fruit, sensor, int_time=[100, 150], led=led,
                                           position=[1, 2, 3, 4], led_current=["25 mA"])
    # x_data, y_data = get_data.get_data(fruit, sensor, int_time=[150], led=led,
    #                                    position=[2], led_current=["25 mA"])

    # y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
    y_data = y_data[y_name]
    print(x_data.shape, y_data.shape)
    # x_scaled_np = StandardScaler().fit_transform(x_data)
    # x_scaled_np = RobustScaler().fit_transform(x_data)
    x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)
    x_scaled_np = StandardScaler().fit_transform(x_scaled_np)
    # x_scaled_np = RobustScaler().fit_transform(x_scaled_np)
    x_data = pd.DataFrame(x_scaled_np, index=x_data.index)
    print(train_index)
    print(test_index)
    print(len(train_index), len(test_index))
    print(x_data.index.unique())
    print(x_data.index.unique().shape)

    X_train = x_data.loc[train_index]
    X_test = x_data.loc[test_index]

    y_train = y_data.loc[train_index]
    y_test = y_data.loc[test_index]
    pls.fit(X_train, y_train)

    y_test_predict = pls.predict(X_test)
    y_train_predict = pls.predict(X_train)
    y_test_predict = pd.DataFrame(y_test_predict, index=y_test.index)
    y_train_predict = pd.DataFrame(y_train_predict, index=y_train.index)

    if average:
        y_test_predict = y_test_predict.groupby(index_name).mean()
        y_train_predict = y_train_predict.groupby(index_name).mean()
        y_test_mean = y_test.groupby(index_name).mean()
        y_train_mean = y_train.groupby(index_name).mean()
    else:
        y_test_mean = y_test
        y_train_mean = y_train

    axis.scatter(y_test_predict, y_test_mean, color='forestgreen', label="Test set")
    axis.scatter(y_train_predict, y_train_mean, color='indigo', label="Training set")
    y_line = [np.min(y_train_predict), np.max(y_train_predict)]
    axis.plot(y_line, y_line, color='red', lw=1, linestyle='--')
    axis.legend(loc='lower right')

    r2_test = r2_score(y_test_mean, y_test_predict)
    mae_test = mean_absolute_error(y_test_mean, y_test_predict)
    r2_train = r2_score(y_train_mean, y_train_predict)
    mae_train = mean_absolute_error(y_train_mean, y_train_predict)
    LEFT_ALIGN = 0.07
    axis.annotate(u"R\u00B2 test ={:.2f}".format(r2_test), xy=(LEFT_ALIGN, 0.82),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    axis.annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    axis.annotate("MAE test ={:.3f}".format(mae_test), xy=(LEFT_ALIGN, 0.66),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    axis.annotate("MAE train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.58),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    # axis.set_ylabel("Measured lycopene (mg/100g)")

    axis.annotate("{0}".format(sensor), xy=(.4, 0.9),
                  xycoords='axes fraction', color='#101028', fontsize='x-large')


figure, ax, = plt.subplots(2, 2, figsize=(8, 8),
                           constrained_layout=True)
figure.suptitle("{0} {1} measurements".format(fruit.capitalize(), y_name), fontsize=24)
# pls = SVR()  # lycopene
make_1figure(pls, "as7262", ax[0][0])
make_1figure(pls, "as7263", ax[0][1])
pls = PLSRegression(n_components=5)
make_1figure(pls, "c12880", ax[1][1])



# pls = TransformedTargetRegressor(regressor=PLSRegression(n_components=8),
#                                  func=neg_log,
#                                  inverse_func=neg_exp)
# pls = TransformedTargetRegressor(regressor=SVR(),
#                                  func=neg_log,
#                                  inverse_func=neg_exp)
pls = TransformedTargetRegressor(regressor=LassoCV(),
                                 func=neg_log,
                                 inverse_func=neg_exp)
# x_data, y_data = get_data.get_data(fruit, "as7265x", int_time=[100, 150], led="b'White IR'",
#                                    position=[1, 2, 3, 4], led_current=["25 mA"])
# y_data = y_data[y_name]
# pls = LassoCV()
# pls.fit(x_data, y_data)
# print(dir(pls))
# print(pls.get_params())
# print(pls.coef_)  # 410, 460, 510, 535, 585, 610, 645, 680
#
# ham
# pls = PLSRegression(n_components=10)
# pls = LassoCV()
# pls = SVR()  # lycopene

make_1figure(pls, "as7265x", ax[1][0], led="b'UV'")


# ax[0][1].set_xlabel("Predicted lycopene (µg/cm²)")

ax[1][0].set_xlabel("Predicted beta-carotene (mg/100g)")
ax[1][1].set_xlabel("Predicted beta-carotene (mg/100g)")
ax[0][0].set_ylabel("Measured beta-carotene (mg/100g)")
ax[1][0].set_ylabel("Measured beta-carotene (mg/100g)")
plt.show()
# figure.savefig(f"{fruit}__4sensors")