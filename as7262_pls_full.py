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
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, PolynomialFeatures
# local files
import get_data
import processing
import sfs

plt.style.use('seaborn')

# fitting_data = pd.read_csv('as7262_roseapple.csv')
# x_data, _, fitting_data = data_get.get_data('as7262 mango', integration_time=150,
#                                             led_current="12.5 mA",
#                                             read_number=2)
# x_data1, _, fitting_data2 = data_get.get_data('as7263 mango', integration_time=150,
#                                               led_current="12.5 mA",
#                                               read_number=2)
x_data, _, fitting_data = get_data.get_data("banana", "as7263", int_time=150,
                                              position=2,
                                              led_current="25 mA", return_type="XYZ")
# x_data2, _, fitting_data2 = data_get.get_data('as7262 mango', integration_time=100,
#                                             led_current="12.5 mA",
#                                             read_number=1)

# x_data = pd.concat([x_data, x_data1], axis=1)
print(x_data)
# ham
# fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()
# fitting_data = fitting_data.drop(["Leaf: 31", "Leaf: 39", "Leaf: 45"])


# fitting_data = fitting_data.loc[(fitting_data['integration time'] == 5) | (fitting_data['integration time'] == 4)]
# fitting_data = fitting_data.loc[(fitting_data['current'] == 25)]
# fitting_data = fitting_data.loc[(fitting_data['position'] == 'pos 2')]
# fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()

exp_transformer = FunctionTransformer(np.exp, inverse_func=np.log)


pls = PLSRegression(n_components=12)
# pls = SGDRegressor()
# pls = DecisionTreeRegressor(max_depth=4)
# pls = GradientBoostingRegressor(max_depth=2)

def invert(x):
    return 1/x

# cachedir = tempfile.mkdtemp()
# mem = Memory(location=cachedir, verbose=1)
# f_regression = mem.cache(feature_selection.f_regression)
# anova = feature_selection.SelectPercentile(f_regression)
# pls = make_pipeline(exp_transformer, pls)
# pls = TransformedTargetRegressor(regressor=PLSRegression(n_components=3),
#                                  func=invert,
#                                  inverse_func=invert)
# pls = TransformedTargetRegressor(regressor=PLSRegression(n_components=3),
#                                  func=np.log,
#                                  inverse_func=np.exp)
#
# pls = LassoCV(max_iter=5000)
# pls = PLSRegression(n_components=4)
# pls = make_pipeline(PLSRegression(n_components=6), LassoCV())
# pls = TransformedTargetRegressor(regressor=GradientBoostingRegressor(),
#                                  func=np.exp,
#                                  inverse_func=np.log)
# pls = GradientBoostingRegressor()
# pls = SVR(kernel="linear")
def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


pls = LinearSVR()
pls = TransformedTargetRegressor(regressor=LinearSVR(),
                                 func=neg_log,
                                 inverse_func=neg_exp)
# rgs = PLSRegression(n_components=6)
#
# pls = TransformedTargetRegressor(regressor=pls,
#                                  func=np.log,
#                                  inverse_func=np.exp)
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
y1 = fitting_data['Avg Total Chlorophyll (µg/cm2)']
y2 = fitting_data['Avg Chlorophyll a (µg/cm2)']
y3 = fitting_data['Avg Chlorophyll b (µg/cm2)']
y4 = y3 / (y2 + y3)



# x_data = fitting_data[x_data_columns]
# x_data, _ = processing.msc(x_data)
# x_data = processing.snv(x_data)
# x_data = np.exp(-x_data)
# x_data = np.log(x_data)
# x_data = 1 / x_data
# scalar = make_pipeline(StandardScaler(), PolynomialFeatures(), KernelPCA(kernel='rbf'))
x_data = StandardScaler().fit_transform(x_data)
x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)

x_scaled = pd.DataFrame(x_scaled_np)
# x_scaled = processing.snv(x_data)

figure, axes, = plt.subplots(2, 2, figsize=(7.5, 8.75), constrained_layout=True)

figure.suptitle("Model fit to new AS7262 Mango data")
# figure.suptitle("Gradient Boosting Regressor fit\nAS7262 Betel data")
axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
invert_y = False
convert_y = False

for i, y in enumerate([y1]):

    bins = np.linspace(y.min(), y.max(), 5)
    labels = ["1", "2", "3", "4"]
    # Y_groups = pd.cut(y, bins)
    # print(Y_groups)
    print('===')
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y,
                                                        test_size=0.2)
    pls.fit(X_train, y_train)
    y_test_predict = pls.predict(X_test)
    y_train_predict = pls.predict(X_train)

    axes[i].scatter(y_test_predict, y_test, color='forestgreen', label="Test set")
    axes[i].scatter(y_train_predict, y_train, color='indigo', label="Training set")
    y_line = [np.min(y_train_predict), np.max(y_train_predict)]
    axes[i].plot(y_line, y_line, color='red', lw=1, linestyle='--')
    axes[i].legend()

    r2_test = r2_score(y_test, y_test_predict)
    # r2_test=1
    mae_test = mean_absolute_error(y_test, y_test_predict)
    # mae_test = 1
    r2_train = r2_score(y_train, y_train_predict)
    # r2_train=1
    mae_train = mean_absolute_error(y_train, y_train_predict)
    # mae_train = 1
    LEFT_ALIGN = 0.07
    axes[i].annotate(u"R\u00B2 test ={:.2f}".format(r2_test), xy=(LEFT_ALIGN, 0.8),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate("MAE test ={:.3f}".format(mae_test), xy=(LEFT_ALIGN, 0.68),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate("MAE train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.62),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].set_title(chloro_columns[i])
    axes[i].set_ylabel("Measured Chlorophyll")
    if i >= 2:
        axes[i].set_xlabel("Predicted Chlorophyll")
    # print(r2_score(y_test, y_test_predict), mean_absolute_error(y_test, y_test_predict))

plt.show()


