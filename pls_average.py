# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import joblib
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split, GroupKFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
# local files
print("pre data")
import get_data
import processing

plt.style.use('seaborn')
print("start")
hplc_fruits = ["mangos", "tomato"]
fruit = "mango"
y_name = 'Avg Total Chlorophyll (µg/cm2)'
index_name = "Leaf number"
AVERAGE = False
if fruit in hplc_fruits:
    index_name = "Fruit"
    y_name = 'lycopene (FW)'
    y_name = "beta-carotene (FW)"
    # y_name = "%DM"
# x_data, y_data = get_data.get_data_as726x_serial(fruit, "as7265x", int_times=[100, 150],
#                                                  positions=[1, 2, 3],
#                                                  led_currents=["25 mA"])
# x_data, y_data = get_data.get_data(fruit, "as7262", int_time=[100, 150], average=False,
#                                    position=[1, 2, 3], led_current=["25 mA"])
# x_data, y_data = get_data.get_data("mango", "as7262", int_time=[150], average=True,
#                                    position=[2], led_current=["25 mA"])
# x_data, y_data = get_data.get_data("rice", "as7265x", int_time=[150], led="b'White UV IR'",
#                                    position=[1, 2, 3], led_current=["12.5 mA"],
#                                    average=False)

df = pd.read_excel("rice_leave_data_w_UT_data.xlsx")
x_columns = []
for column in df.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = df[x_columns]
y_data = df["Avg Total Chlorophyll (µg/cm2)"]
print(y_data)

# x_data, y_data = get_data.get_data(fruit, "as7262", int_time=[100, 150], average=False,
#                                    position=[1, 2, 3], led_current=["25 mA"])
# x_data, y_data = get_data.get_data(fruit, "c12880", int_time=None, average=False,
#                                    position=None, led_current=None)
print(y_data)

print(x_data)
# x_data = x_data.iloc[:, :12]
# x_data = x_data.loc[y_data[y_name] > 35]
# y_data = y_data.loc[y_data[y_name] > 35]
print('==', x_data.shape)

# print(y_data.to_string())
print('======')

# y_data = y_data[y_name]
# plt.scatter(np.arange(len(y_data)), y_data)
# plt.show()
print(x_data.shape, y_data.shape)


# x_data_filter = savgol_filter(x_data, 5, polyorder=2, deriv=0)
# x_data = pd.DataFrame(x_data_filter, index=x_data.index)
# x_data = processing.snv(x_data)

# x_scaled_np = StandardScaler().fit_transform(x_data)
# x_data = pd.DataFrame(x_scaled_np, index=x_data.index)
# x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_data)
# x_scaled_np = StandardScaler().fit_transform(x_scaled_np)
#
# x_data = pd.DataFrame(x_scaled_np, index=x_data.index)


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    # print(-np.log(x))
    # for i, num in enumerate(-np.log(x)):
    #     print(i, num)
    #
    # print(-np.log(x))
    return -np.log(x)


def inverse(x):
    return 1/x

# cv = GroupKFold(n_splits=5)
cv = ShuffleSplit(n_splits=100)
# pls = PLSRegression(n_components=8)
# x_data = -np.log(1/x_data)
# pls = AdaBoostRegressor()
# pls = RandomForestRegressor(n_estimators=50, max_depth=4)
# pls = SVR(kernel='linear', degree=1)
# pls = SVR()
# pls = LassoCV(cv=cv)
# pls = make_pipeline(pls, Lasso())
# pls = make_pipeline(PLSRegression(n_components=12),
#                     RandomForestRegressor(n_estimators=5))
# pls = TransformedTargetRegressor(regressor=pls,
#                                  func=neg_log,
#                                  inverse_func=neg_exp)
# pls = TransformedTargetRegressor(regressor=pls,
#                                  func=inverse,
#                                  inverse_func=inverse)


# X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,
#                                                     test_size=0.2)
print(x_data.shape)
print(x_data.index)
print('--=====----===--===--')
print(x_data.index.unique())
train_index, test_index, _, _ = train_test_split(x_data.index.unique(), x_data.index.unique(),
                                                 test_size=0.2)
print('l', train_index)
print('p', test_index)

print(x_data.to_string)

print(len(train_index), len(test_index))
print(train_index)
X_train = x_data.loc[train_index]
X_test = x_data.loc[test_index]

y_train = y_data.loc[train_index]
y_test = y_data.loc[test_index]
print('+++', X_train.shape, X_test.shape)
print(y_train)
print(X_train)
pls.fit(X_train, y_train)
y_test_predict = pls.predict(X_test)
y_train_predict = pls.predict(X_train)
y_test_predict = pd.DataFrame(y_test_predict, index=y_test.index)
y_train_predict = pd.DataFrame(y_train_predict, index=y_train.index)

print(y_test_predict)

if AVERAGE:
    y_test_predict = y_test_predict.groupby(index_name).mean()
    y_train_predict = y_train_predict.groupby(index_name).mean()

    y_test_mean = y_test.groupby(index_name).mean()
    y_train_mean = y_train.groupby(index_name).mean()
else:
    y_test_mean = y_test
    y_train_mean = y_train

print(y_test_predict)
print('-----++', y_test_predict.shape, y_train_predict.shape, index_name)
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
# joblib.dump(pls, 'rice_random_forest_full_chloro.joblib')
if hasattr(pls, "coef_"):
    print(pls.coef_)
print(dir(pls))

plt.annotate(u"R\u00B2 test ={:.2f}".format(r2_test), xy=(LEFT_ALIGN, 0.8),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.annotate("MAE test ={:.3f}".format(mae_test), xy=(LEFT_ALIGN, 0.68),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.annotate("MAE train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.62),
                 xycoords='axes fraction', color='#101028', fontsize='large')
plt.title(y_name)
plt.ylabel("Measured Chlorophyll")
plt.ylabel("Predicted Chlorophyll")

# plt.figure(2)
# plt.bar()
plt.show()

