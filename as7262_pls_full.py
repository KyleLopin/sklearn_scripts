# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
# local files
import data_getter
import processing

plt.style.use('seaborn')

# fitting_data = pd.read_csv('as7262_roseapple.csv')
x_data, _, fitting_data = data_getter.get_data('as7262 betal')
fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()
# fitting_data = fitting_data.drop(["Leaf: 17"])

print(fitting_data.columns)
# fitting_data = fitting_data.loc[(fitting_data['integration time'] == 5) | (fitting_data['integration time'] == 4)]
# fitting_data = fitting_data.loc[(fitting_data['current'] == 25)]
# fitting_data = fitting_data.loc[(fitting_data['position'] == 'pos 2')]
# fitting_data = fitting_data.groupby('Leaf number', as_index=True).mean()

pls = PLSRegression(n_components=4)
# pls = DecisionTreeRegressor(max_depth=5)

x_data_columns = []
for column in fitting_data:
    if 'nm' in column:
        x_data_columns.append(column)

chloro_columns = ['Total Chlorophyll (ug/ml)', 'Chlorophyll a (ug/ml)',
                  'Chlorophyll b (ug/ml)', "Fraction Chlorophyll b"]

y1 = fitting_data['Total Chlorophyll (ug/ml)']
y2 = fitting_data['Chlorophyll a (ug/ml)']
y3 = fitting_data['Chlorophyll b (ug/ml)']
y4 = y3 / (y2 + y3)


x_data = fitting_data[x_data_columns]
# x_data = processing.snv(x_data)
# x_data = np.exp(x_data)
# x_data = 1 / x_data

x_scaled_np = StandardScaler().fit_transform(x_data)

x_scaled = pd.DataFrame(x_scaled_np, columns=x_data.columns)

figure, axes, = plt.subplots(2, 2, figsize=(7.5, 8.75), constrained_layout=True)
figure.suptitle("PLS 12 components of AS7263 Roseapple data")
axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
invert_y = True
for i, y in enumerate([y1, y2, y3, y4]):
    if invert_y:
        y = 1 / y
    # print(1/y, 1/(1/y))
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y,
                                                        test_size=0.33,
                                                        random_state=2)

    pls.fit(X_train, y_train)
    y_test_predict = pls.predict(X_test)
    y_train_predict = pls.predict(X_train)

    if invert_y:
        y_train = 1 / y_train
        y_test = 1 / y_test
        y_test_predict = 1 / y_test_predict
        y_train_predict = 1 / y_train_predict

    # y = 1 / y
    # y_predict = 1 / y_predict
    # x_linespace = np.linspace(np.min(y_train_predict), np.max(y_train_predict))
    # print(x_linespace)
    # y_linespace = pls.predict(x_linespace.reshape(-1, 1))

    axes[i].scatter(y_test_predict, y_test, color='forestgreen', label="Test set")
    axes[i].scatter(y_train_predict, y_train, color='indigo', label="Training set")
    y_line = [np.min(y_train_predict), np.max(y_train_predict)]
    axes[i].plot(y_line, y_line, color='red', lw=1, linestyle='--')
    axes[i].legend()
    # axes[i].scatter(1/y, 1/(1/y))
    r2_test = r2_score(y_test, y_test_predict)
    mae_test = mean_absolute_error(y_test, y_test_predict)
    r2_train = r2_score(y_train, y_train_predict)
    mae_train = mean_absolute_error(y_train, y_train_predict)
    LEFT_ALIGN = 0.07
    axes[i].annotate(u"R\u00B2 test ={:.2f}".format(r2_test), xy=(LEFT_ALIGN, 0.8),
                  xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate(u"R\u00B2 train ={:.2f}".format(r2_train), xy=(LEFT_ALIGN, 0.74),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate("MAE test ={:.3f}".format(mae_test), xy=(LEFT_ALIGN, 0.68),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].annotate("MEA train ={:.3f}".format(mae_train), xy=(LEFT_ALIGN, 0.62),
                     xycoords='axes fraction', color='#101028', fontsize='large')
    axes[i].set_title(chloro_columns[i])
    print(r2_score(y_test, y_test_predict), mean_absolute_error(y_test, y_test_predict))

plt.show()


