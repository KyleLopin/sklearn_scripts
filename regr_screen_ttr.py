# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.model_selection import cross_validate, RepeatedKFold

# local files
import get_data
import full_regrs
import processing


sensor = "as7262"
leaf = "mango"

filename = f"{sensor}_{leaf}_results.xlsx"


def neg_exp(x):
    return np.exp(-x) + 1


def neg_log(x):
    print(x)
    print('x:', -np.log(x))
    return -np.log(x)


print(neg_exp(5))
print(neg_log(neg_exp(5)))

x, y = get_data.get_data("mango", "as7262", int_time=150,
                             position=2,
                             led_current="25 mA")
y = y['Total Chlorophyll (µg/cm2)']

# ttr = TransformedTargetRegressor(regressor=PLS(n_components=3),
#                                  func=neg_exp, inverse_func=neg_log)
ttr = TransformedTargetRegressor(regressor=PLS(n_components=3),
                                 func=neg_exp, inverse_func=neg_log)
ttr.fit(x, y)
ttr.score(x, y)
ham

ttr_funcs = [(neg_exp, neg_log), ]

all_regressors = full_regrs.get_all_regrs()
all_transformers = full_regrs.get_transformers()
cv = RepeatedKFold(n_splits=4, n_repeats=15)

training_scores = []
test_scores = []
regressors = []
df_columns = []
for trns in all_transformers.keys():
    for score_set in [" train", " test"]:
        df_columns.append(trns+score_set)


def run_scan(x, y, sheet_name):
    results_df = pd.DataFrame([], index=all_regressors.keys(), columns=df_columns)
    for tr_name, transform in all_transformers.items():
        print('tr: ', tr_name, sheet_name)
        print(x.shape)
        print(y.shape)
        x_tr = transform.fit_transform(x)
        for name, regrs in all_regressors.items():
            print(name, tr_name)
            regressors.append(name)
            try:
                scores = cross_validate(regrs, x_tr, y, cv=cv,
                                        scoring=('r2', 'neg_mean_absolute_error'),
                                        return_train_score=True)
                r2_scores = [scores['train_r2'].mean(),
                             scores['test_r2'].mean()]
                mae_scores = [scores['train_neg_mean_absolute_error'].mean(),
                              scores['test_neg_mean_absolute_error'].mean()]
                print(r2_scores)
                print(mae_scores)
                training_scores.append(mae_scores[0])
                test_scores.append(mae_scores[1])
                results_df[tr_name+" train"][name] = mae_scores[0]
                results_df[tr_name + " test"][name] = mae_scores[1]
            except:
                pass

    with pd.ExcelWriter(filename, mode='a') as writer:
        results_df.to_excel(writer, sheet_name=sheet_name)
