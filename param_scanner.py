# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model
from sklearn.model_selection import cross_validate, GroupShuffleSplit, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# local files
import data_get

plt.style.use('seaborn')

x_data, _y, full_data = data_get.get_data('as7262 mango', average=False)

print(full_data.columns)
currents = full_data['LED current'].unique()
times = full_data['integration time'].unique()
print(currents, times)
print(full_data['saturation check'].unique())
pls = PLSRegression(n_components=6)
# pls = linear_model.LinearRegression()
cv = RepeatedKFold(n_splits=5, n_repeats=20)
cv_group = GroupShuffleSplit(n_splits=200)
scores = []
labels = []
errors = []
training_scores = []
training_errors = []
for current in currents:
    for time in times:
        print(current, time)
        labels.append("{0}, {1}".format(current, time))

        X, Y = data_get.get_data("as7262 mango", integration_time=time,
                                 led_current=current, return_type="XY",
                                 read_number=1)
        print(X.shape)
        X = StandardScaler().fit_transform(X)
        X = PolynomialFeatures(degree=2).fit_transform(X)
        # print(X)
        # print('S', X.isnull().values.any())
        # print(Y.columns)
        Y = Y['Total Chlorophyll (µg/cm2)']
        # print(Y)
        # print('Y', Y.isnull().values.any())
        bins = np.linspace(Y.min(), Y.max(), 5)
        labels = ["1", "2", "3", "4"]
        # print(bins)
        Y_groups = pd.cut(Y, bins, labels=labels)
        # print(Y_groups)
        # print('groups', Y_groups.isnull().values.any())
        scores_ = cross_validate(pls, X, Y, cv=cv, groups=Y_groups,
                                 scoring='neg_mean_absolute_error',
                                 return_train_score=True)
        print(scores_)
        print(scores_['test_score'].mean(), scores_['test_score'].std())
        print(scores_['train_score'].mean(), scores_['train_score'].std())
        scores.append(scores_['test_score'].mean())
        errors.append(scores_['test_score'].std())
        training_scores.append(scores_['train_score'].mean())
        training_errors.append(scores_['train_score'].std())

x = np.arange(len(scores))
width = 0.4
plt.title("Multiple reads\nAS7262 Mango leaves")
plt.bar(x, scores, width=width, yerr=errors)
plt.bar(x+width, training_scores, width=width, yerr=training_errors)
plt.show()
