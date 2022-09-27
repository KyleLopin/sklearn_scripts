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
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, PolynomialFeatures

# local files
import get_data

plt.style.use('seaborn')

fruit = "sugacane"
cv = GroupKFold(n_splits=5)
train_index, test_index, _, _ = train_test_split(np.arange(1, 101), np.arange(1, 101),
                                                 test_size=0.2)


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


pls = PLSRegression(n_components=6)
pls = TransformedTargetRegressor(regressor=pls,
                                 func=neg_log,
                                 inverse_func=neg_exp)


def make_1figure(sensor, axis, led=None):
    x_data, y_data = get_data.get_data(fruit, sensor, int_time=[100, 150], led=led,
                                       position=[1, 2, 3], led_current=["25 mA"])
    # x_data, y_data = get_data.get_data(fruit, sensor, int_time=[150], led=led,
    #                                    position=[2], led_current=["25 mA"])

    y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
    print(x_data.shape, y_data.shape)
    # x_data = StandardScaler().fit_transform(x_data)
    x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)
    x_scaled_np = StandardScaler().fit_transform(x_scaled_np)
    x_data = pd.DataFrame(x_scaled_np, index=x_data.index)

    X_train = x_data.loc[train_index]
    X_test = x_data.loc[test_index]

    y_train = y_data.loc[train_index]
    y_test = y_data.loc[test_index]
    pls.fit(X_train, y_train)

    y_test_predict = pls.predict(X_test)
    y_train_predict = pls.predict(X_train)
    y_test_predict = pd.DataFrame(y_test_predict, index=y_test.index)
    y_train_predict = pd.DataFrame(y_train_predict, index=y_train.index)

    y_test_predict = y_test_predict.groupby("Leaf number").mean()
    y_train_predict = y_train_predict.groupby("Leaf number").mean()
    y_test_mean = y_test.groupby("Leaf number").mean()
    y_train_mean = y_train.groupby("Leaf number").mean()

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
    axis.set_ylabel("Measured Chlorophyll (µg/cm²)")

    axis.annotate("{0} sensor".format(sensor), xy=(.4, 0.9),
                  xycoords='axes fraction', color='#101028', fontsize='x-large')


figure, ax, = plt.subplots(3, 1, figsize=(7, 11),
                           constrained_layout=True)
figure.suptitle("{0} leave chlorophyll measurements".format(fruit.capitalize()), fontsize=24)
make_1figure("as7262", ax[0])
make_1figure("as7263", ax[1])
make_1figure("as7265x", ax[2], led="b'White'")

ax[2].set_xlabel("Predicted Chlorophyll (µg/cm²)")
plt.show()
figure.savefig(f"{fruit}_3sensors")
