# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
from sklearn.model_selection import cross_val_score


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

