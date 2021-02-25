# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   RobustScaler, StandardScaler)

# local files
import get_data
import full_regrs
import processing


sensor = "as7262"
sensors = ["as7262", "as7263"]
leafs = ["mango", "rice", "jasmine", "banana", "sugarcane"]
leafs = ["banana"]

filename = f"ini_screen_results.xlsx"
# with pd.ExcelWriter(filename, mode='w') as writer:
#     pd.DataFrame().to_excel(writer)

def neg_exp(x):
    return np.exp(-x)


def neg_log(x):
    return -np.log(x)


# fig, (ax1, ax2) = plt.subplots(1, 2)
# mms = MinMaxScaler()
#
# x = np.arange(100)
# # x_tran = mms.fit_transform(x.reshape(-1, 1))
# print(x)
# y = np_exp(x)
# print(y)
# y_inverse = np_log(y)
# # y_inverse = mms.inverse_transform(y_inverse.reshape(-1, 1))
# ax1.plot(x, y_inverse, 'k')
# ax1.plot(x, x, 'r')
# ax2.plot(x, y)
# plt.show()

# ham


x, y = get_data.get_data("mango", "as7262", int_time=150,
                             position=2,
                             led_current="25 mA")
y = y['Total Chlorophyll (µg/cm2)']
# x =

# ttr = TransformedTargetRegressor(regressor=PLS(n_components=3),
#                                  func=np.square, inverse_func=np.sqrt)
# ttr = TransformedTargetRegressor(regressor=PLS(n_components=3),
#                                  func=neg_exp, inverse_func=neg_log)
# ttr.fit(x, y)
# ttr.score(x, y)
# ham

ttr_funcs = [("None"),
             ("Square", np.square, np.sqrt),
             ("Square root", np.sqrt, np.square),
             ("Log", neg_log, neg_exp)]
# ttr_funcs = [[("None")]]

all_regressors = full_regrs.get_all_regrs()
all_transformers = full_regrs.get_transformers()
cv = RepeatedKFold(n_splits=4, n_repeats=15)

training_scores = []
test_scores = []
regressors = []
df_columns = []
for trns in all_transformers.keys():
    for score_set in [" train", " test"]:
        for ttr_func in ttr_funcs:
            df_columns.append(trns+" "+ttr_func[0]+score_set)


def run_scan(x, y, sheetname):
    results_df = pd.DataFrame([], index=all_regressors.keys(), columns=df_columns)
    for tr_name, transform in all_transformers.items():
        # print('tr: ', tr_name)
        # print(x.shape)
        # print(y.shape)
        x_tr = transform.fit_transform(x)
        for name, regr in all_regressors.items():

            for ttr in ttr_funcs:

                if len(ttr) == 2:
                    # single func
                    ttr_regr = TransformedTargetRegressor(regressor=regr,
                                                          transformer=ttr[1])
                elif len(ttr) == 3:
                    ttr_regr = TransformedTargetRegressor(regressor=regr,
                                                          func=ttr[1],
                                                          inverse_func=ttr[2])
                elif len(ttr) == 1:
                    ttr_regr = regr

                print(name, tr_name, ttr[0], sheetname)
                regressors.append(name)
                try:
                    scores = cross_validate(ttr_regr, x_tr, y, cv=cv,
                                            scoring=('r2', 'neg_mean_absolute_error'),
                                            return_train_score=True)
                    r2_scores = [scores['train_r2'].mean(),
                                 scores['test_r2'].mean()]
                    mae_scores = [scores['train_neg_mean_absolute_error'].mean(),
                                  scores['test_neg_mean_absolute_error'].mean()]
                    # print(r2_scores)
                    # print(mae_scores)
                    training_scores.append(mae_scores[0])
                    test_scores.append(mae_scores[1])
                    print(mae_scores[1], tr_name+" "+ttr[0]+" train")
                    results_df[tr_name+" "+ttr[0]+" train"][name] = mae_scores[0]
                    results_df[tr_name+" "+ttr[0]+" test"][name] = mae_scores[1]
                except:
                    pass

    with pd.ExcelWriter(filename, mode='a') as writer:
        results_df.to_excel(writer, sheet_name=sheetname)


if __name__ == "__main__":
    for sensor in sensors:
        for leaf in leafs:
            x, y = get_data.get_data(leaf, sensor, int_time=150,
                                     position=2,
                                     led_current="25 mA")
            print(y.columns)
            y_column = 'Total Chlorophyll (µg/cm2)'
            if y_column not in y.columns:
                y_column = 'Avg Total Chlorophyll (µg/cm2)'
            y = y[y_column]
            name = f"{sensor}_{leaf}"
            run_scan(x, y, name)
