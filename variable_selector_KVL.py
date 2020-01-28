# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import pickle
# installed libraries
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


data_full = pd.read_csv('wildbetal_flouro_rows.csv')
# data_full = data_full[data_full['position'] == 'pos 2']
y_name = 'Chlorophyll a (ug/ml)'
y_data = data_full[y_name]


# y_data = data_full['Chlorophyll a (ug/ml)'] / data_full['Chlorophyll b (ug/ml)']

# print(data_full)
x_data_columns = []
for column in data_full:
    if 'nm' in column:
        x_data_columns.append(column)

x_data = data_full[x_data_columns]
print(x_data)
print(type(x_data))
print(x_data.shape)
print(y_data.shape)


class ModelFit(object):
    def __init__(self):
        self.test_score = []
        self.test_stdev = []
        self.train_score = []
        self.train_stdev = []


def pls_variable_selection(x, y, num_pls_components):
    """

    Adopted from https://nirpyresearch.com/variable-selection-method-pls-python/
    :param x:
    :param y:
    :param max_components:
    :param scorer:
    :return:
    """
    # initialize new model parameter holder
    scores = dict()
    scores['r2'] = ModelFit()
    scores['mae'] = ModelFit()
    cut_conditions = []
    num_varaibles = []
    # make a score table to fill in
    # scores = np.zeros( x.shape[1] )
    # print('==========')
    pls = PLSRegression(num_pls_components)
    usable_columns = None
    best_score = 0
    x_scaled_np = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_np, columns=x.columns)
    # print(x_scaled)
    while x_scaled.shape[1] >= num_pls_components:
        print('shape: ', x_scaled.shape, num_pls_components, best_score)
        number_to_cut = int(x_scaled.shape[1] / 100)
        if number_to_cut == 0:
            number_to_cut = 1
        # print('number to cut: ', number_to_cut, num_pls_components)
        pls.fit(x_scaled, y)
        # y_predict = pls.predict(x_scaled)
        # score = r2_score(y, y_predict)
        cv_splitter = 3  # passing to corss_validate will implement a KFold with 3 folds
        group_splitter = None
        if x_scaled.shape[1] <= 200:
            # cv_splitter = ShuffleSplit(n_splits=100, test_size=0.35)
            cv_splitter = GroupShuffleSplit(n_splits=100, test_size=0.35)
            group_splitter = data_full['Leaf number']
        elif x_scaled.shape[1] <= 400:
            # cv_splitter = ShuffleSplit(n_splits=30, test_size=0.35)
            cv_splitter = GroupShuffleSplit(n_splits=30, test_size=0.35)
            group_splitter = data_full['Leaf number']
        local_scores = cross_validate(pls, x_scaled, y, cv=cv_splitter,
                                      return_train_score=True, groups=group_splitter,
                                      scoring=['r2', 'neg_mean_absolute_error'])

        scores['r2'].train_score.append(local_scores['train_r2'].mean())
        scores['r2'].train_stdev.append(local_scores['train_r2'].std())
        scores['r2'].test_score.append(local_scores['test_r2'].mean())
        scores['r2'].test_stdev.append(local_scores['test_r2'].std())

        scores['mae'].train_score.append(local_scores['train_neg_mean_absolute_error'].mean())
        scores['mae'].train_stdev.append(local_scores['train_neg_mean_absolute_error'].std())
        scores['mae'].test_score.append(local_scores['test_neg_mean_absolute_error'].mean())
        scores['mae'].test_stdev.append(local_scores['test_neg_mean_absolute_error'].std())

        num_varaibles.append(x_scaled.shape[1])
        if scores['r2'].test_score[-1] > best_score:
            best_score = scores['r2'].test_score[-1]
            usable_columns = x_scaled.columns


        # print(pls.coef_[:, 0])
        # print(pls.coef_.shape)
        sorted_coeff = np.argsort( np.abs(pls.coef_[:, 0]) )
        # print('1')
        # print(sorted_coeff)
        # print( pls.coef_[:, 0][sorted_coeff] )
        # print('2')
        # print(sorted_coeff[-5:])
        # print(sorted_coeff[-1])
        # print(x_scaled)
        # print(pls.coef_[:, 0][sorted_coeff[-1]], pls.coef_[:, 0][sorted_coeff[0]])
        # print(sorted_coeff[-1], x_scaled.columns[sorted_coeff[0]])
        # print(scores['r2'].train_score[-1], scores['r2'].test_score[-1],
        #       scores['mae'].train_score[-1], scores['mae'].test_score[-1])
        # column_to_drop = x_scaled.columns[sorted_coeff[0]]
        columns_to_drop = x_scaled.columns[sorted_coeff[:number_to_cut]]
        # print(columns_to_drop.values)
        if x_scaled.shape[1] < 50:
            # print('dropping: ', columns_to_drop)
            # print(columns_to_drop.values)
            cut_conditions.append(columns_to_drop.values)

        x_scaled.drop(columns=columns_to_drop, inplace=True)

    # print(usable_columns)
    # print('===========')
    # print(x_scaled.columns)
    # print(cut_conditions)
    # data = dict()
    # data['test means'] = test_scores_average
    # data['test std'] = test_scores_std
    # data['train means'] = train_scores_average
    # data['train std'] = train_scores_std
    # data['columns'] = usable_columns
    # data['num variables'] = num_varaibles
    scores['columns'] = usable_columns
    scores['num variables'] = num_varaibles
    # print('========')
    # print(data.keys())
    # filename = "param_selector_{0}.pickle".format(num_pls_components)
    # with open(filename, 'wb') as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return scores


def parameter_scanner(x, y):
    for i in range(39, 80):
        print('running i =', i)
        print('========================')
        data_i = pls_variable_selection(x, y, i)
        filename = "ch_a_param_selector_{0}.pickle".format(i)
        with open(filename, 'wb') as f:
            pickle.dump(data_i, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    y = np.array([1, 2, 3])
    x = np.array([[1, 2, 3], [10, 20, 30], [15, 27, 38]])
    # cv_splitter = GroupShuffleSplit(n_splits=100,
    #                                 test_size=0.35,
    #                                 random_state=120)
    group_splitter = data_full['Leaf number']
    # pls_variable_selection(x_data, y_data, 22)
    parameter_scanner(x_data, y_data)
