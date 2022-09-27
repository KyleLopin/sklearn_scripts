# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import (cross_val_predict, train_test_split, cross_validate,
                                     LeaveOneGroupOut, GroupShuffleSplit)
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# local files
import get_data
import sfs

plt.style.use('seaborn')

x_data, y_data = get_data.get_data("mango", "as7262", int_time=[150],
                                   position=[1, 2, 3], led_current=["25 mA"])

print(x_data)
print('==')
print(y_data.to_string())
print('======')

y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
print(x_data.shape, y_data.shape)
# x_data_ss = StandardScaler().fit_transform(x_data)
poly_feat = PolynomialFeatures(degree=2)
x_scaled_np = poly_feat.fit_transform(x_data)

# x_scaled_np = StandardScaler().fit_transform(x_scaled_np)
x_data = pd.DataFrame(x_scaled_np, index=x_data.index,
                      columns=poly_feat.get_feature_names())


def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


pls = PLSRegression(n_components=6)
pls = TransformedTargetRegressor(regressor=pls,
                                 func=neg_log,
                                 inverse_func=neg_exp)
cv = LeaveOneGroupOut()
cv_ss = GroupShuffleSplit(n_splits=100)

def base_pls_cv(x, y, n_comps, return_model=False):
    pls_base = PLSRegression(n_components=n_comps)
    ttr_pls = TransformedTargetRegressor(regressor=pls_base,
                                         func=neg_log,
                                         inverse_func=neg_exp)
    y_cv = cross_val_predict(ttr_pls, x, y, cv=cv, groups=x.index)
    # y_cv = cross_val_predict(ttr_pls, x, y, cv=cv)
    score = r2_score(y, y_cv)
    rmsecv = mean_absolute_error(y, y_cv)
    if return_model == False:
        return (y_cv, score, rmsecv)
    else:
        return (y_cv, score, rmsecv, ttr_pls)


def pls_optimize_comps(x, y, max_comps):
    test_scores = np.zeros(max_comps)
    training_scores = np.zeros(max_comps)
    for i in range(1, max_comps+1, 1):
        pls_base = PLSRegression(n_components=i)
        ttr_pls = TransformedTargetRegressor(regressor=pls_base,
                                             func=neg_log,
                                             inverse_func=neg_exp)
        # ttr_pls.fit(x, y)
        # y_cv = cross_val_predict(ttr_pls, x, y, cv=cv, groups=x.index)
        scores = cross_validate(ttr_pls, x, y, cv=cv_ss, return_train_score=True,
                                groups=x.index, scoring="neg_mean_absolute_error")
        # score = r2_score(y, y_cv)
        test_scores[i-1] = scores['test_score'].mean()
        training_scores[i-1] = scores['train_score'].mean()
        print('i = ', i, test_scores[i-1])
    opt_comps, max_score = np.argmax(test_scores), test_scores[np.argmax(test_scores)]
    return (opt_comps+1, max_score, test_scores, training_scores)


def get_max_score(sfs_scores):
    print(sfs_scores)
    max_cv_score = max(sfs_scores["test scores"])
    max_score_index = sfs_scores["test scores"].index(max_cv_score)
    best_columns = sfs_scores["columns"][max_score_index]
    print(max_cv_score)
    print(max_score_index)
    print(best_columns)
    ham


def pls_optimize_n_comps_with_sfs(x, y, max_comps):
    test_scores = np.zeros(max_comps)
    training_scores = np.zeros(max_comps)
    for i in range(1, max_comps + 1, 1):
        pls_base = PLSRegression(n_components=i)
        ttr_pls = TransformedTargetRegressor(regressor=pls_base,
                                             func=neg_log,
                                             inverse_func=neg_exp)
        sfs_scores = sfs.sfs_back(ttr_pls, x, y, cv_ss, group=x.index)
        print(sfs_scores)
        print('=====')


if __name__ == "__main__":
    # y_cv, score, rmsecv = base_pls_cv(x_data, y_data, 5)
    #
    # print(y_cv)
    # print(score)
    # print(rmsecv)
    # print('=====', x_data.shape)
    # opt_comps, min_score, test_sc, train_sc = pls_optimize_comps(x_data, y_data, 11)
    # print(opt_comps)
    # print(min_score)
    # print(x_data)
    pls_optimize_n_comps_with_sfs(x_data, y_data, 12)
    # plt.plot(test_sc, 'r')
    # plt.plot(train_sc, 'blue')
    # plt.show()
