# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import pickle
# installed libraries
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as sfs_plot
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (learning_curve,
    RepeatedKFold, ShuffleSplit, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
# local files
import get_data
import sfs


# cv_stats = RepeatedKFold(n_splits=4, n_repeats=50)
cv_stats = ShuffleSplit(n_splits=100, test_size=.25)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, group=None,
                        n_jobs=None, ax=None, train_sizes=np.linspace(.4, 1.0, 3)):
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
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # print(test_scores_mean, train_scores_mean)
    # print(test_scores_std, train_scores_std)
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # ax.legend(loc="best")
    ax.grid(True)
    plt.show()

    # return plt



def pls_screen_as726x(x, y, n_comps=8):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    pls = PLSRegression(n_components=n_comps)
    lasso = MultiTaskLassoCV(max_iter=40000)

    regr = make_pipeline(PolynomialFeatures(), pls)
    # regr = make_pipeline(PolynomialFeatures(), lasso)
    plot_learning_curve(regr, "Learning Curve", x, y, ax=ax2)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    print('1')
    plt.show()


def pls_comp_screen(x, y, max_comps=6):
    data_fits = dict()
    for i in range(1, max_comps-1):
        print('n_comps = ', i)
        run_fit = pls_sfs_screen(x, y, n_comps=i)
        data_fits[i] = run_fit
    with open("sfs_test_degree1.pkl", 'wb') as f:
        pickle.dump(data_fits, f)


def pls_sfs_screen(x, y, n_comps=6):
    cv = RepeatedKFold(n_splits=4, n_repeats=25)
    print(x.shape)
    regr = PLSRegression(n_components=n_comps)
    # sfs = SequentialFeatureSelector(regr,
    #                                 k_features=(max_comps, 2*max_comps),
    #                                 forward=False,
    #                                 floating=False,
    #                                 scoring='neg_mean_absolute_error',
    #                                 cv=cv, verbose=2)
    # sfs_fit = sfs.fit(x, y)
    sfs_fit = sfs.sfs_full(regr, x, y, cv, min_components=n_comps)
    # print(sfs.subsets_)
    # print('========')
    # print(sfs.get_metric_dict())
    # print('+++++++')
    # cv_scores = []
    # test_scores = []
    # with open("sfs_test.pkl", 'wb') as f:
    #     pickle.dump(sfs_fit, f)
    # for key, data in sfs_fit.items():
    #     print(data)
    #     print(key)
        # cv_scores.append(data['avg_score'])
        # test_scores.append(sfs_fit['mean_test_score'])
    print('--------')
    # print(cv_scores)
    # sfs_plot(sfs.get_metric_dict(), kind='std_dev')
    return sfs_fit


def find_maxfit(data):
    max_score = 100
    print(data["training scores"])
    return max(data["test scores"]), max(data["training scores"])


if __name__ == "__main__":
    x, y = get_data.get_data("mango", "as7262", int_time=150,
                             position=2,
                             led_current="25 mA")

    pls = PLSRegression(n_components=6)
    y = y['Total Chlorophyll (µg/cm2)']
    # x, y = get_data.get_data("mango", "as7262", int_time=150,
    #                          position=2, led="b'White'",
    #                          led_current="25 mA")
    # print(x.shape)
    # # pls_screen_as726x(x, y, n_comps=10)
    # print(type(x))
    poly = PolynomialFeatures(degree=1)
    x_trans = poly.fit_transform(x)
    # pls.fit(x_trans, y)
    # y_predict = pls.predict(x_trans)
    # print(mean_absolute_error(y, y_predict))
    # ham
    # n_comps = 6
    # regr = PLSRegression(n_components=n_comps)
    # print(x_trans.shape)
    # print(poly.get_feature_names())
    #
    x_trans = pd.DataFrame(x_trans, columns=poly.get_feature_names())
    print(x_trans)
    cols_to_use = []
    for column in poly.get_feature_names():
        if ' ' not in column:
            cols_to_use.append(column)
    print(cols_to_use)
    x_trans = x_trans[cols_to_use]
    print(x_trans)
    # svr = SVR()
    # pls = PLSRegression(n_components=6)
    # regr = pls
    # print(y.columns)
    # # pls.fit(x, y['Avg Total Chlorophyll (µg/cm2)'])
    # # print(pls.coef_)
    # # plot_learning_curve(pls, "", x_trans, y['Avg Total Chlorophyll (µg/cm2)'])
    # # ham
    # pls_sfs_screen(x_trans, y)
    pls_comp_screen(x_trans, y, max_comps=6)

    with open("sfs_test_degree1.pkl", 'rb') as f:
        sfs_fit = pickle.load(f)
    # sfs_fit.pop(13, None)
    n_comps = []
    best_score = []
    training = []
    for key, value in sfs_fit.items():
        print('n comp:', key)
        print(value)
        if not value['test scores']:
            print('breaking')
            continue
        # find_maxfit(value)
        n_comps.append(key)
        test, train = find_maxfit(value)
        print(test, train)
        best_score.append(test)
        training.append(train)

        # print(best_score)
    plt.plot(n_comps, best_score, 'k-')
    plt.plot(n_comps, training, 'r-')
    # # plt.plot(sfs_fit["n columns"], sfs_fit['test scores'], color='green')
    # # plt.plot(sfs_fit["n columns"], sfs_fit['training scores'], color='red')
    print("Done")
    plt.show()
