# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import tempfile
from joblib import Memory


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import feature_selection
from sklearn.linear_model import LassoCV, SGDRegressor
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
# local files
import data_get
import processing
import sfs

plt.style.use('seaborn')

# fitting_data = pd.read_csv('as7262_roseapple.csv')
x_data, _, fitting_data = data_get.get_data('as7262 mango', integration_time=150,
                                            led_current="12.5 mA", average=False)
# x_data, _, fitting_data = data_get.get_data('as7262 mango', integration_time=200,
#                                             led_current="12.5 mA",
#                                             read_number=2)
# x_data1, _, fitting_data2 = data_get.get_data('as7262 mango', integration_time=50,
#                                             led_current="12.5 mA",
#                                             read_number=2)
# x_data2, _, fitting_data3 = data_get.get_data('as7262 mango', integration_time=150,
#                                             led_current="25 mA",
#                                             read_number=2)
#
# x_data = pd.concat([x_data, x_data1, x_data2], axis=1)
# print(fitting_data.columns)
# x_data["Leaf number"] = fitting_data['Leaf number']

# fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()
# fitting_data = fitting_data.drop(["Leaf: 31", "Leaf: 39", "Leaf: 45"])
# fitting_data = fitting_data.loc[(fitting_data['integration time'] == 5) | (fitting_data['integration time'] == 4)]

exp_transformer = FunctionTransformer(np.exp, inverse_func=np.log)


pls = PLSRegression(n_components=6)
# pls = SGDRegressor()
# pls = DecisionTreeRegressor(max_depth=4)
# pls = GradientBoostingRegressor(max_depth=2)

def invert(x):
    return 1/x

# pls = make_pipeline(exp_transformer, pls)
# pls = TransformedTargetRegressor(regressor=PLSRegression(n_components=3),
#                                  func=invert,
#                                  inverse_func=invert)
# pls = TransformedTargetRegressor(regressor=PLSRegression(n_components=3),
#                                  func=np.log,
#                                  inverse_func=np.exp)
#
pls = LassoCV(max_iter=5000)
pls = PLSRegression(n_components=6)
# pls = make_pipeline(PLSRegression(n_components=6), LassoCV())
# pls = TransformedTargetRegressor(regressor=GradientBoostingRegressor(),
#                                  func=np.exp,
#                                  inverse_func=np.log)
# pls = GradientBoostingRegressor()
# pls = SVR(kernel='rbf')
# pls = TransformedTargetRegressor(regressor=rgs,
#                                  func=np.exp,
#                                  inverse_func=np.log)
# pls = SVR()
x_data_columns = []
for column in fitting_data:
    if 'nm' in column:
        x_data_columns.append(column)

chloro_columns = ['Total Chlorophyll (µg/mg)', 'Chlorophyll a (µg/mg)',
                  'Chlorophyll b (µg/mg)', "Fraction Chlorophyll b"]
# print(fitting_data[chloro_columns])
print(fitting_data.columns)
# y1 = fitting_data['Total Chlorophyll (µg/mg)']
y1 = fitting_data['Total Chlorophyll (µg/cm2)']
y2 = fitting_data['Chlorophyll a (µg/cm2)']
y3 = fitting_data['Chlorophyll b (µg/cm2)']
# y3 = fitting_data['Chlorophyll a (µg/mg)']
y4 = y3 / (y2 + y3)
# y4 = fitting_data['Chlorophyll a (µg/mg)']
# y2 = fitting_data['Total Chlorophyll (µg/mg)']
# y3 = fitting_data['Total Chlorophyll (µg/mg)']
# y4 = fitting_data['Total Chlorophyll (µg/mg)']


x_data = fitting_data[x_data_columns]
# x_data, _ = processing.msc(x_data)
# x_data = processing.snv(x_data)
# x_data = np.exp(-x_data)
# x_data = np.log(x_data)
# x_data = 1 / x_data
# scalar = make_pipeline(StandardScaler(), PolynomialFeatures(), KernelPCA(kernel='rbf'))
x_scaled_np = StandardScaler().fit_transform(x_data)
x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_scaled_np)

# x_scaled_np = PCA(n_components=12).fit_transform(x_scaled_np)
# x_scaled_np = KernelPCA(kernel='rbf', n_components=12).fit_transform(x_scaled_np)

print('====')

# print(x_scaled_np)
# x_scaled_np = PLSRegression(n_components=12).fit_transform(x_scaled_np)
print(x_scaled_np)
# x_scaled_np = scalar.fit_transform(x_data)
# x_scaled = pd.DataFrame(x_scaled_np, columns=x_data.columns)
x_scaled = pd.DataFrame(x_scaled_np)
# x_scaled = x_data
print(x_scaled)
# x_scaled['Leaf number'] = fitting_data['Leaf number']
figure, axes, = plt.subplots(2, 2, figsize=(7.5, 8.75), constrained_layout=True)

figure.suptitle("Model fit to new AS7262 Mango data\nmultiple reads")
# figure.suptitle("Gradient Boosting Regressor fit\nAS7262 Betel data")
axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
invert_y = False
convert_y = False

for i, y in enumerate([y1, y2, y3, y4]):
    if convert_y:
        y = np.exp(-y)
    if invert_y:
        y = 1 / y

    # X_train, X_test, y_train, y_test = train_test_split(x_scaled, y,
    #                                                     test_size=0.2)
    #
    # pls.fit(X_train, y_train)
    # y_test_predict = pls.predict(X_test)
    # y_train_predict = pls.predict(X_train)
    # print(y)
    # print(pls.coef_)
    loo = LeaveOneOut()
    loo = LeaveOneGroupOut()
    print(x_scaled)
    # group = x_scaled['Leaf number']
    y_predict = []
    y_actual = []
    y_train_predict = []
    y_train_actual = []

    for train_index, test_index in loo.split(x_scaled, groups=fitting_data["Leaf number"]):
        # for train_index, test_index in loo.split(x_scaled):
        X_train, X_test = x_scaled_np[train_index], x_scaled_np[test_index]
        print(y.to_numpy())
        y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
        pls.fit(X_train, y_train)

        x_scaled_np = sfs.sfs_back()

        y_train_predict = pls.predict(X_train)

        # ham

        # y_predict.append(pls.predict(X_test))
        y_actual.append(y_test.mean())
        y_predict.append(pls.predict(X_test).mean())

        # y_train_actual.append()
        # y_train_predict.append(mean_absolute_error(y_train_predict, y_train))

        # y_actual

    if invert_y:
        y_train = 1 / y_train
        y_test = 1 / y_test
        y_test_predict = 1 / y_test_predict
        y_train_predict = 1 / y_train_predict
    if convert_y:
        y_train = -np.log(y_train)
        y_test = -np.log(y_test)
        y_test_predict = -np.log(y_test_predict)
        y_train_predict = -np.log(y_train_predict)

    # y = 1 / y
    # y_predict = 1 / y_predict
    # x_linespace = np.linspace(np.min(y_train_predict), np.max(y_train_predict))
    # print(x_linespace)
    # y_linespace = pls.predict(x_linespace.reshape(-1, 1))

    # axes[i].scatter(y_test_predict, y_test, color='forestgreen', label="Test set")
    axes[i].scatter(y_predict, y_actual, color='indigo', label="Training set")
    y_line = [np.min(y_predict), np.max(y_predict)]
    axes[i].plot(y_line, y_line, color='red', lw=1, linestyle='--')
    axes[i].legend()
    # axes[i].scatter(1/y, 1/(1/y))
    # print(y_test)
    # print(y_test_predict)
    # print(X_test)
    # print(pls.regressor_.coef_)
    print(y_actual)
    print(y_predict)

    r2_test = r2_score(y_actual, y_predict)
    # r2_test=1
    mae_test = mean_absolute_error(y_actual, y_predict)
    # mae_test = 1
    r2_train = r2_score(y_actual, y_predict)
    # r2_train=1
    mae_train = mean_absolute_error(y_actual, y_predict)
    # mae_train = 1
    LEFT_ALIGN = 0.07
    axes[i].annotate(u"R\u00B2 test ={:.2f}".format(r2_test), xy=(LEFT_ALIGN, 0.8),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    # axes[i].annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
    #                  xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate("MAE test ={:.3f}".format(mae_test), xy=(LEFT_ALIGN, 0.68),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    # axes[i].annotate("MAE train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.62),
    #                  xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].set_title(chloro_columns[i])
    axes[i].set_ylabel("Measured Chlorophyll")
    if i >= 2:
        axes[i].set_xlabel("Predicted Chlorophyll")
    # print(r2_score(y_test, y_test_predict), mean_absolute_error(y_test, y_test_predict))

plt.show()


