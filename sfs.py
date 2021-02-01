# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate


def sfs_back(est, X_original, Y, cv, penalty_rate=0.002):
    best_score = -np.inf
    best_columns = []
    current_column_set = X_original.columns
    X = X_original.copy()

    while X.shape[1] > 3:
        print('i = ', X.shape[1])
        print(current_column_set)
        x_current = X[current_column_set]

        best_scan_score = -np.inf
        best_scan_columns = []

        for j, new_column in enumerate():

            x_new = x_current.copy()
            x_new.drop(new_column)

            score = cross_val_score(est, x_new, Y, cv=cv, scoring='neg_mean_absolute_error').mean()
            penalty = penalty_rate * x_new.shape[1]
            print(score)
            if score - penalty > best_scan_score:
                best_scan_score = score - penalty
                print('new score: ', best_scan_score)
                best_scan_columns = x_new.columns
                print('best scan columns: ', best_scan_columns)

        if best_scan_score > best_score:
            best_score = best_scan_score
            print('new score: ', best_scan_score)
            best_columns = best_scan_columns
            print('best scan columns: ', best_columns)


def sfs_full(regr, X_original, y, cv, min_components=4, penalty_rate=0.0):
    cv_scores = []
    test_scores = []
    current_column_set = X_original.columns
    X = X_original.copy()
    while X.shape[1] > min_components:
        print('i = ', X.shape[1])
        # print(current_column_set)
        x_current = X[current_column_set]

        best_scan_score = -np.inf
        best_scan_train_score = 0
        best_scan_columns = []
        for j, new_column in enumerate(x_current.columns):
            # print(j, new_column)
            x_new = x_current.copy()
            x_new.drop(new_column, axis=1)
            # print(x_new.columns)
            # score = cross_val_score(regr, x_new, y, cv=cv, scoring='neg_mean_absolute_error').mean()
            scores = cross_validate(regr, x_new, y, cv=cv,
                                    scoring='neg_mean_absolute_error',
                                    return_train_score=True)
            penalty = penalty_rate * x_new.shape[1]
            # print(scores)
            # print(scores.keys())
            cv_score = scores['test_score'].mean()
            train_score = scores['train_score'].mean()
            if cv_score - penalty > best_scan_score:
                best_scan_score = cv_score - penalty
                best_scan_train_score = train_score
                best_scan_columns = x_new.columns
                print('new score: ', best_scan_score, best_scan_train_score, len(best_scan_columns))

                # print('best scan columns: ', best_scan_columns)
