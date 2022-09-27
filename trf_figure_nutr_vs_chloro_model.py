# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import clone
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
# fruit = "tomato"
# y_name = "lycopene (FW)"

hplc_fruits = ["mangoes", "tomato"]
index_name = "Leaf number"
average = True
if fruit in hplc_fruits:
    index_name = "Fruit"
    # y_name = 'lycopene (FW)'
    y_name = "beta-carotene (FW)"
    # y_name = "%DM"
    y_name = "phenols (FW)"
    # y_name = "carotene (FW)"

waves = ['410 nm', '435 nm', '460 nm', '485 nm',
         '510 nm', '535 nm', '565 nm']
waves = ['460 nm', '485 nm',
         '510 nm', '535 nm']
waves = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm',
         '535 nm', '565 nm', '585 nm', '610 nm', '645 nm',
         '680 nm', '705 nm', '730 nm', '760 nm', '810 nm',
         '860 nm', '900 nm', '940 nm']
# waves = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm', '565 nm']
cv = GroupKFold(n_splits=5)
# lycopene
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=6)
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=13)
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=5)
# train_index, test_index, _, _ = train_test_split(np.arange(1, 61), np.arange(1, 61),
#                                                  test_size=0.2, random_state=8)


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


pls = PLSRegression(n_components=1)
# pls = TransformedTargetRegressor(regressor=pls,
#                                  func=neg_log,
#                                  inverse_func=neg_exp)


def make_1figure(pls, sensor, axis, led=None):
    # make chloro model
    if sensor == "c12880":
        x_data, y_data = get_data.get_data("mango", sensor)
    else:
        x_data, y_data = get_data.get_data("mango", sensor, int_time=[100, 150], led=led,
                                           position=[1, 2, 3, 4], led_current=["25 mA"])
    x_data = x_data[waves]
    y_chl_data = y_data['Avg Total Chlorophyll (µg/cm2)']
    x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)
    x_scaled_np = StandardScaler().fit_transform(x_scaled_np)
    # x_scaled_np = RobustScaler().fit_transform(x_scaled_np)
    x_data = pd.DataFrame(x_scaled_np, index=x_data.index)
    chloro_model = clone(pls)
    chloro_model.fit(x_data, y_chl_data)


    # make nutrient model

    if sensor == "c12880":
        x_data, y_data = get_data.get_data(fruit, sensor)
    else:
        x_data, y_data = get_data.get_data(fruit, sensor, int_time=[100, 150], led=led,
                                           position=[1, 2, 3, 4], led_current=["25 mA"])
    # print(y_data.columns)
    y_data = y_data[y_name]
    x_data_waves = x_data.copy()
    x_data_waves = x_data_waves[waves]
    print(x_data.shape, y_data.shape)
    # x_scaled_np = StandardScaler().fit_transform(x_data)
    # x_scaled_np = RobustScaler().fit_transform(x_data)
    x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)
    x_scaled_np = StandardScaler().fit_transform(x_scaled_np)
    x_scaled_np_w = PolynomialFeatures(degree=2).fit_transform(x_data_waves)
    x_scaled_np_w = StandardScaler().fit_transform(x_scaled_np_w)
    # x_scaled_np = RobustScaler().fit_transform(x_scaled_np)
    x_data = pd.DataFrame(x_scaled_np, index=x_data.index)
    x_data_waves = pd.DataFrame(x_scaled_np_w, index=x_data_waves.index)

    pls.fit(x_data, y_data)

    y_predict = pls.predict(x_data)
    y_predict = pd.DataFrame(y_predict, index=y_data.index)
    chloro_predict = chloro_model.predict(x_data_waves)
    chloro_predict = pd.DataFrame(chloro_predict, index=y_data.index)

    if average:
        y_mean = y_predict.groupby(index_name).mean()
        chloro_mean = chloro_predict.groupby(index_name).mean()
        y_mean_chloro = y_data.groupby(index_name).mean()
    chloro_mean = (100 - chloro_mean) / 3.5
    chloro_mean = chloro_mean.values.flatten()
    # axis.scatter(chloro_mean, y_mean, color='forestgreen', label="Test set")
    axis.scatter(chloro_mean, y_mean_chloro,
                 color='purple', label="Test set")

    # axis.scatter(y_train_predict, y_train_mean, color='indigo', label="Training set")
    # y_line = [np.min(y_mean), np.max(y_mean)]
    # axis.plot(y_line, y_line, color='red', lw=1, linestyle='--')
    # axis.legend(loc='upper right')
    # print(chloro_mean.values.flatten())
    # r2 = r2_score(y_mean, chloro_mean)
    # mae = mean_absolute_error(y_mean, chloro_mean)
    fit = np.polyfit(chloro_mean, y_mean_chloro.values.flatten(), 1)
    print(fit.tolist())
    corr = np.corrcoef(chloro_mean, y_mean_chloro.values.flatten())[0, 1]**2
    print(corr)
    # ham
    p = np.poly1d(fit)
    x_line = np.arange(np.min(chloro_mean), np.max(chloro_mean))
    axis.plot(x_line, p(x_line), 'r--')
    # r2_train = r2_score(y_train_mean, y_train_predict)
    # mae_train = mean_absolute_error(y_train_mean, y_train_predict)
    LEFT_ALIGN = 0.8
    axis.annotate(u"R\u00B2 ={:.2f}".format(corr), xy=(LEFT_ALIGN, 0.78),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    # # axis.annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
    # #               xycoords='axes fraction', color='#101028', fontsize='large')
    # axis.annotate("MAE ={:.3f}".format(mae), xy=(LEFT_ALIGN, 0.66),
    #               xycoords='axes fraction', color='#101028', fontsize='large')
    # axis.annotate("MAE train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.58),
    #               xycoords='axes fraction', color='#101028', fontsize='large')
    # # axis.set_ylabel("Measured lycopene (mg/100g)")
    #
    # axis.annotate("{0}".format(sensor), xy=(.4, 0.9),
    #               xycoords='axes fraction', color='#101028', fontsize='x-large')


figure, ax = plt.subplots(1, 1, figsize=(8, 8),
                           constrained_layout=True)
figure.suptitle("{0} {1}\nvs chlorophyll relationship".format(fruit.capitalize(), y_name), fontsize=24)
# pls = SVR()  # lycopene
# make_1figure(pls, "as7262", ax[0][0])
# make_1figure(pls, "as7263", ax[0][1])
pls = PLSRegression(n_components=5)
# make_1figure(pls, "c12880", ax[1][1])
print(ax)

# ham


# pls = TransformedTargetRegressor(regressor=PLSRegression(n_components=8),
#                                  func=neg_log,
#                                  inverse_func=neg_exp)
# pls = TransformedTargetRegressor(regressor=SVR(),
#                                  func=neg_log,
#                                  inverse_func=neg_exp)
# pls = TransformedTargetRegressor(regressor=LassoCV(),
#                                  func=neg_log,
#                                  inverse_func=neg_exp)
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

make_1figure(pls, "as7265x", ax, led="b'White IR'")


# ax[0][1].set_xlabel("Predicted lycopene (µg/cm²)")

ax.set_xlabel("Modeled chlorophyll (µg/cm²)")

ax.set_ylabel(f"{y_name} (mg/100g)")
# ax[1][0].set_ylabel("Measured beta-carotene (mg/100g)")
plt.show()
# figure.savefig(f"{fruit}__4sensors")
