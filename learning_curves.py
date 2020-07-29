# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# local files
import data_getter

plt.style.use('seaborn')


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
                        n_jobs=None, train_sizes=np.linspace(.6, 1.0, 4)):
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
    plt.figure(figsize=(5, 4))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        groups=group, shuffle=True, scoring='r2')
    # explained_variance, neg_median_absolute_error
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    return plt


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

if __name__ == '__main__':
    pls = PLS(n_components=12)
    ridge = Ridge(random_state=0, max_iter=5000)
    svr = SVR(C=100)
    lasso = Lasso(max_iter=5000, alpha=10**-4)
    x_data, _, data = data_getter.get_data('as7263 roseapple verbose')
    data = data.groupby('Leaf number', as_index=True).mean()

    y_name = 'Total Chlorophyll (ug/ml)'
    y_data = ( 1 / data[y_name])
    # x_data = x_data.groupby('Leaf number', as_index=True).mean()

    # x_data = data

    cv = ShuffleSplit(n_splits=100, test_size=0.333, random_state=0)

    plot_learning_curve(pls, 'AS7263 Roseapple 12-component PLS Learning Curve\n' \
                                   'Total Chlorophyll', x_data, y_data, cv=cv)

plt.show()