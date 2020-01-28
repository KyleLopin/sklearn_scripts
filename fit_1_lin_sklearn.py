# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Use SKLearn module to build a linear model using test / validation data sets
"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import median_absolute_error, r2_score
from sklearn import linear_model


plt.style.use('seaborn')
refl_data = pd.read_csv('as7262_mango.csv')

# refl_data = refl_data.loc[(refl_data['position'] == 'pos 3')]
# refl_data = refl_data.groupby('Leaf number', as_index=True).mean()
data_columns = []
for column in refl_data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)
chloro_columns = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                  'Total Chlorophyll (ug/ml)']

# x_data = refl_data[data_columns]
x_data = refl_data[['550 nm']]
print(x_data)
y_data = refl_data['Total Chlorophyll (ug/ml)']
# y_data = 1/ y_data

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


_linear_model = linear_model.LinearRegression()

model_fit = _linear_model.fit(x_train, y_train)
test_prediction = _linear_model.predict(x_test)

print(model_fit)
print(test_prediction)
print(x_test.shape, y_test.shape)
print(test_prediction.shape, y_test.shape)
# plt.scatter(x_test, y_test)
# plt.plot(x_test, test_prediction, 'r')
print(u'r\u00B2 = {0:.2f}\nMAE = {1:.4f}'.format(model_fit.score(x_test, y_test),
                                     median_absolute_error(y_test, test_prediction)))
# plt.show()

loo = LeaveOneOut()
loo.get_n_splits(x_data)

print(x_data)
x_data
MEA = []
R2 = []
for train_index, test_index in loo.split(x_data):
    # print(train_index)
    # print(test_index)
    x_train, x_test = x_data.loc[train_index], x_data.loc[test_index]
    y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
    model_fit = _linear_model.fit(x_train, y_train)
    test_prediction = _linear_model.predict(x_test)
    MEA_loo = median_absolute_error(y_test, test_prediction)
    MEA.append(MEA_loo)
    R2 = model_fit.score(x_test, y_test)
    print(u'r\u00B2 = {0:.2f}\nMAE = {1:.4f}'.format(R2, MEA_loo))

print(MEA)
print(np.array(MEA).mean())
