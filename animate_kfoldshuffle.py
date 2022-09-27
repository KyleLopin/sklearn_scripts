# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold

import get_data
plt.style.use('ggplot')

x, y, fitting_data = get_data.get_data("mango", "as7265x", int_time=150,
                                       position=2, average=False, led="b'White IR'",
                                       led_current="25 mA", return_type="XYZ")
print(y)
y = y['Avg Total Chlorophyll (µg/cm2)']
x = x.to_numpy()
y = y.to_numpy()
print(y.shape)
regr = LinearRegression()
cv = RepeatedKFold(n_splits=5, n_repeats=10)
fig = plt.figure(figsize=(8, 8))

train_splits = []
test_splits = []
for train_index, test_index in cv.split(x):
    train_splits.append(train_index)
    test_splits.append(test_index)

print(len(train_splits))
train_plt, test_plt = None, None
text1, text2, text3, text4 = None, None, None, None


def runner(i):
    global train_plt, test_plt, text1, text2, text3, text4
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # regr.fit(X_train, y_train)
    train_index = train_splits[i]
    test_index = test_splits[i]

    X_train = x[train_index]
    y_train = y[train_index]
    X_test = x[test_index]
    y_test = y[test_index]

    regr.fit(X_train, y_train)

    y_predict_train = regr.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_predict_train)
    y_predict_test = regr.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_predict_test)

    if i == 0:
        plt.title("AS7265X fits to mango leaves chlorophyll levels\nLinear Regression", size=18)
        plt.plot([0, 90], [0, 90], c='mediumvioletred', ls='--')
        plt.xlabel("Measured chlorophyll (µg/cm\u00B2)", size=15)
        plt.ylabel("Predicted chlorophyll (µg/cm\u00B2)", size=15)
        plt.legend(loc="lower right", prop={'size': 18})

    if train_plt:
        train_plt.remove()
        test_plt.remove()

    if text1:
        text1.remove()
        text2.remove()
        text3.remove()
        text4.remove()

    train_plt = plt.scatter(y_train, y_predict_train, color='indigo', label="Training set")
    test_plt = plt.scatter(y_test, y_predict_test, color='forestgreen', label="Test set")

    text1 = plt.annotate("R\u00B2 training set ={:.2f}".format(regr.score(X_train, y_train)), xy=(.05, 0.94),
                 xycoords='axes fraction', color='#101028', fontsize='large')
    text2 = plt.annotate("Mean absolute error training set = {:.2f}".format(mae_train), xy=(.05, 0.88),
                 xycoords='axes fraction', color='#101028', fontsize='large')

    text3 = plt.annotate("R\u00B2 training set ={:.2f}".format(regr.score(X_test, y_test)), xy=(.05, 0.82),
                 xycoords='axes fraction', color='#101028', fontsize='large')
    text4 = plt.annotate("Mean absolute error training set = {:.2f}".format(mae_test), xy=(.05, 0.76),
                 xycoords='axes fraction', color='#101028', fontsize='large')


time = range(50)
ani = FuncAnimation(fig, runner, blit=False, interval=500,
                    frames=time, repeat=False)
# plt.show()
ani.save("KFoldRepeat.gif", writer='imagemagick')
