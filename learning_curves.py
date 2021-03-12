# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, LinearRegression
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.svm import SVR, LinearSVR

# local files
import data_get
import get_data
import processing

plt.style.use('seaborn')


def invert(x):
    return 1/x

# x_data_columns = []
# leds = {}
# for column in data_full:
#     print(column)
#     if 'nm' in column:
#         x_data_columns.append(column)
#         print(column)
#         led = column.split(',')[1].strip()
#         print(led)
#         if led not in leds:
#             leds[led] = []
#         leds[led].append(column)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, group=None,
                        n_jobs=None, ax=None, train_sizes=np.linspace(.6, 1.0, 4)):
    """
    from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if not ax:
        figure, ax, = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

    # ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # ax.set_xlabel("Training examples")
    # ax.set_ylabel("Negative Mean Absolute Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        groups=group, shuffle=True, scoring='neg_mean_absolute_error')
    # explained_variance, neg_median_absolute_error
    print(test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    print(test_scores_mean, train_scores_mean)
    print(test_scores_std, train_scores_std)
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # ax.legend(loc="best")
    ax.grid(True)
    # plt.show()

    # return plt


def mango_refl3_linear():
    data = pd.read_csv('mango_chloro_refl3.csv')
    print(data.columns)
    # data = data.loc[(data['position'] == 'pos 1') | (data['position'] == 'pos 2')]
    data = data.loc[(data['integration time'] == 200)]
    # data = data.groupby('Leaf number', as_index=True).mean()
    X = data[['450 nm', '500 nm', '550 nm', '570 nm', '600 nm']].values
    print(X)
    print(type(X))
    print(X.shape)
    print(X.dtype)
    print(np.unique(X))
    print('-====')

    y = data[['Total Chlorophyll (ug/ml)']].values
    y = 1/y
    print(y)
    print(type(y))
    print(y.shape)
    print(y.dtype)

    title = "Learning Curves (Linear Regression)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # estimator = SVR()
    estimator = linear_model.LinearRegression()
    plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4)

    title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # estimator = SVR(gamma=0.001)
    # plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()


def pls_learning(num_components, num_vars):
    print(1)

    data_full = pd.read_csv('mango_flouro_rows.csv')
    data_full = data_full[data_full['position'] == 'pos 2']
    y_name = 'Total Chlorophyll (ug/ml)'
    # y_name = 'Chlorophyll b (ug/ml)'
    y_data = data_full[y_name]

    x_data = data_full[x_data_columns]

    x_data = get_best_pls_variables(x_data, y_data,
                                    num_components, num_vars)


    cv = GroupShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
    group_splitter = data_full['Leaf number']
    estimator = PLS(num_components)
    title = "Learning curve {} components".format(num_components)

    plot_learning_curve(estimator, title, x_data, y_data,
                        cv=cv, group=group_splitter)
    plt.show()


def led_columns(leds):
    pass


def get_best_pls_variables(x, y, num_pls_components,
                           num_varaibles):
    x_scaled_np = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_np, columns=x.columns)

    pls = PLS(num_pls_components)
    pls.fit(x_scaled, y)
    sorted_coeff = np.argsort(np.abs(pls.coef_[:, 0]))
    sorted_coeff = np.flip(sorted_coeff)
    columns_to_keep = x.columns[sorted_coeff[:num_varaibles]]
    print(columns_to_keep)
    return x_scaled[columns_to_keep]


def plot_4_learning_curves(estimators, est_names, cv, X, Y, ylim=None):
    figure, axes, = plt.subplots(2, 2, figsize=(7.5, 8.75), constrained_layout=True)

    figure.suptitle("Model fit to new AS7262 Mango data")
    # figure.suptitle("Gradient Boosting Regressor fit\nAS7262 Betel data")
    axes_ = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    print(axes_)
    for i, ax in enumerate(axes_):
        print('i=', i, ax)
        plot_learning_curve(estimators[i], est_names[i], X, Y,
                            cv=cv, ax=ax, ylim=ylim)
    plt.show()


def plot_param_learning_curves():
    x_data, _y, full_data = data_get.get_data('as7262 mango', average=False)
    pls = PLS(n_components=6)
    print(full_data.columns)
    currents = full_data['LED current'].unique()
    times = full_data['integration time'].unique()
    print(currents, times)
    print(full_data['saturation check'].unique())
    figure, axes, = plt.subplots(len(currents), len(times),
                                 figsize=(9, 12),
                                 constrained_layout=True)

    figure.suptitle("Parameter scan of new AS7262 Mango data")
    # figure.suptitle("Gradient Boosting Regressor fit\nAS7262 Betel data")
    # axes_ = [axes[0][0], axes[0][1], axes[0][2], axes[0][3],
    #          axes[1][0], axes[1][1], axes[1][2], axes[1][3],
    #          axes[2][0], axes[2][1], axes[2][2], axes[2][3],
    #          axes[3][0], axes[3][1], axes[3][2], axes[3][3],
    #          axes[4][0], axes[4][1], axes[4][2], axes[4][3],]

    current_i = 0
    time_i = 0
    for current in currents:
        for time in times:
            X, Y = data_get.get_data("as7262 mango", integration_time=time, led_current=current, return_type="XY")

            X = StandardScaler().fit_transform(X)
            X = PolynomialFeatures().fit_transform(X)

            Y = Y['Total Chlorophyll (µg/mg)']
            title = str(time*2.8)+" ms "+current
            print(title)

            plot_learning_curve(pls, title, X, Y,
                                cv=cv, ax=axes[current_i][time_i], ylim=[-0.3, -.1])

            time_i += 1
        time_i = 0
        current_i += 1


if __name__ == '__main__':
    pls = PLS(n_components=6)
    ridge = Ridge(random_state=0, max_iter=20000)
    svr = SVR(C=10)
    lasso = Lasso(max_iter=5000, alpha=10**-4)
    gradboost = GradientBoostingRegressor(max_depth=2)
    lr = LinearRegression()


    def neg_exp(x):
        return np.exp(-x)


    def neg_log(x):
        return -np.log(x)


    # pls = LinearSVR()
    pls = TransformedTargetRegressor(regressor=LinearSVR(max_iter=2000000),
                                     func=neg_log,
                                     inverse_func=neg_exp)
    # x_data, _, data = data_get.get_data('as7262 mango', led_current="25 mA",
    #                                     integration_time=100)
    x_data, _, data = get_data.get_data("mango", "as7262", int_time=150,
                                        position=2, led_current="25 mA",
                                        return_type="XYZ")
    # x_data, _ = processing.msc(x_data)
    # x_data = processing.snv(x_data)
    x_data = StandardScaler().fit_transform(x_data)
    # x_data = PolynomialFeatures(degree=2).fit_transform(x_data)
    # x_data = RobustScaler().fit_transform(x_data)
    # x_data, _, data = data_getter.get_data('new as7262 mango')
    # data = data.groupby('Leaf number', as_index=True).mean()

    y_name = 'Avg Total Chlorophyll (µg/cm2)'
    y_data = (data[y_name])
    # y_data = np.exp(-y_data)
    # x_data = x_data.groupby('Leaf number', as_index=True).mean()

    # x_data = data

    cv = ShuffleSplit(n_splits=100, test_size=0.2)

    plot_learning_curve(pls, 'AS7262 Mango Learning Curve\n'
                                   'Total Chlorophyll', x_data, y_data, cv=cv)
    # estimators = [lasso, svr, pls, lr]
    # est_names = ["Lasso", "SVR", "PLS", "Linear Regression"]
    # plot_4_learning_curves(estimators, est_names, cv, x_data, y_data, [-15, 0])
    # plot_param_learning_curves()

    plt.show()
