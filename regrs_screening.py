# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.pipeline import make_pipeline

# local files
import data_getter
import full_regrs
import processing

filename = "as7262 betal results.xlsx"
x_data, _y, data = data_getter.get_data('as7262 ylang')

all_regressors = full_regrs.get_all_regrs()
all_transformers = full_regrs.get_transformers()
print(all_regressors)

print(x_data)

chloro_types = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                'Total Chlorophyll (ug/ml)']

_y = _y['Total Chlorophyll (ug/ml)']
print(_y)
cv = RepeatedKFold(n_splits=4, n_repeats=15)

transformers = []

training_scores = []
test_scores = []
regressors = []
df_columns = []
for trns in all_transformers.keys():
    for score_set in [" train", " test"]:
        df_columns.append(trns+score_set)
results_df = pd.DataFrame([], index=all_regressors.keys(), columns=df_columns)
print(results_df)

def run_scan(X, y, sheet_name):
    results_df = pd.DataFrame([], index=all_regressors.keys(), columns=df_columns)
    print(results_df)

    for tr_name, transform in all_transformers.items():
        print('tr: ', tr_name, sheet_name)
        print(X.shape)
        print(y.shape)
        X_tr = transform.fit_transform(X)

        for name, regrs in all_regressors.items():
            print(name, tr_name)
            regressors.append(name)

            # model = regrs.fit(x_data, y)
            # score = model.score(x_data, y)
            # print(name, ' : ', score)
            try:
                scores = cross_validate(regrs, X_tr, y, cv=cv,
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


ys = {"Normal Y": _y, "Inverse Y": 1/_y, "Log Y": np.log(_y),
      "Inverse Log Y": np.log(1/_y), "Exp Y": np.exp(-_y),
      }
ys = {"Normal Y": _y}
mcs_x, _ = processing.msc(x_data)
print(mcs_x)

Xs = {"Normal X": x_data, "SNV": processing.snv(x_data)}

# Xs = {"MSC": mcs_x}
with pd.ExcelWriter(filename, mode='w') as writer:
    results_df.to_excel(writer)
for y_name, y_inner in ys.items():
    for x_name, x_inner in Xs.items():
        new_sheet = x_name + ' ' + y_name
        run_scan(x_inner, y_inner, new_sheet)

print(regressors)
print(training_scores)
print(test_scores)
print(results_df)
results_df.to_csv("as7262_betal_results.csv")
