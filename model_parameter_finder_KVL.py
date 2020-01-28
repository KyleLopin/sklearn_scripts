# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import pickle
# installed libraries
import numpy as np
import pandas as pd
# import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def regression_model(y, x):
    ones = np.ones(len(x))
    x = np.column_stack((x, ones))
    x = sm.add_constant(x)
    model = sm.OLS(_y, x)
    results = model.fit()
    return results


def fit_2_vars(data_columns, extra_columns=None):
    print(extra_columns)
    for i, data_column1 in enumerate(data_columns):
        for j, data_column2 in enumerate(data_columns):
            if j <= i:
                continue
            print('1: ', data_column1, data_column2)
            # x = data[[data_column1, data_column2]]
            pre_x = [data_column1, data_column2]
            pre_x.extend(extra_columns)
            print(pre_x)
            x = data[pre_x]

            print(x)
            print(data_column1, data_column2)


# 'Total Chlorophyll (ug/ml)', 'Chlorophyll a (ug/ml)'
class KVLModelParameterFinder():
    def __init__(self, dataframe: pd.DataFrame, data_columns=None,
                 y_dataname="Total Chlorophyll (ug/ml)", n_splits=5):
        self.dataframe = dataframe
        self.max_iterations = len(dataframe.columns)
        self.y_data = dataframe[y_dataname]
        self.cross_validator = KFold(n_splits=n_splits)
        self.linear_model = linear_model.LinearRegression()
        self.lowest_qualifing_score = 0
        self.scores_list = []
        self.score_stds = []
        self.train_score_list = []
        self.train_stds = []
        self.details_list = []
        self.debugger = []
        self.full_data_set = dict()
        self.number_top_scores_to_save = 5
        self.max_iterations = 5
        print(self.max_iterations)
        print(dataframe.shape)
        pre_data_columns = dataframe.columns
        self.data_columns = []
        for _data_column in pre_data_columns:
            if ' nm' in _data_column:
                self.data_columns.append(_data_column)

        print(data_columns)
        extra_data = ['680 nm, White LED, int: 150', '860 nm, 395 nm LED, int: 250']

        self.find_n_more_parameters(8)

        # self.fit_one_more_column(extra_data)

    def find_n_more_parameters(self, num_params):
        next_column = []
        self.full_data_set[0] = dict()
        self.full_data_set[0]['details'] = [[]]
        for i in range(1, num_params+1):
            self.full_data_set[i] = dict()
            print('i start =', i)
            print(self.full_data_set[i-1])
            for prev_fit in self.full_data_set[i-1]['details']:
                print(prev_fit, type(prev_fit))

                scores, stds, train_score, train_stds, details = self.fit_one_more_column(prev_fit)
                print('retur: ', i)
                self.full_data_set[i]['test scores'] = scores
                self.full_data_set[i]['test stds'] = stds
                self.full_data_set[i]['train scores'] = train_score
                self.full_data_set[i]['train stds'] = train_stds
                self.full_data_set[i]['details'] = details
                for key, value in self.full_data_set.items():
                    print('key: ', key)
                    print(value)

        # save the data in a pickle
        with open('full_data_set_fit.pickle', 'wb') as pkl:
            pickle.dump(self.full_data_set, pkl)

    def fit_one_more_column(self, prefit_columns=[]):
        print('fiting with: ', prefit_columns)
        prefit_data = self.dataframe[prefit_columns]
        self.scores_list = []
        self.score_stds = []
        self.details_list = []
        self.lowest_qualifing_score = 0
        # print(prefit_data)
        for i, data_column in enumerate(self.data_columns):

            if i > 10:
                continue
            # print(' i = ', i)
            # print(i, data_column)
            fitting_data = pd.concat([prefit_data, self.dataframe[data_column]], axis=1)
            # print(fitting_data)
            self.cross_validator.get_n_splits(fitting_data)
            cv_scores = []
            training_scores = []
            for train_index, test_index in self.cross_validator.split(fitting_data):
                x_train, x_test = fitting_data.loc[train_index], fitting_data.loc[test_index]
                y_train, y_test = self.y_data.loc[train_index], self.y_data.loc[test_index]
                model_fit = self.linear_model.fit(x_train, y_train)
                score = model_fit.score(x_test, y_test)
                training_score = model_fit.score(x_train, y_train)
                training_scores.append(training_score)
                cv_scores.append(score)
            cv_scores = np.array(cv_scores)
            training_scores = np.array(training_scores)
            print('cv_mean = ', cv_scores.mean(), self.lowest_qualifing_score)
            # print(training_scores)
            if cv_scores.mean() > self.lowest_qualifing_score:
                self.scores_list.append(cv_scores.mean())
                self.score_stds.append(cv_scores.std())

                self.train_score_list.append(training_scores.mean())
                self.train_stds.append(training_scores.std())

                # print('==========', type(self.details_list))
                # print(prefit_columns, data_column)
                # print(type(prefit_columns), type(data_column))
                # print('1bb:', prefit_columns + [data_column], type(self.details_list))
                self.details_list.append(prefit_columns + [data_column])
                # print(self.scores_list)
                self.debugger.append((cv_scores.mean(), cv_scores.std(), self.details_list[-1]))
        print('0000000000: ', len(self.scores_list))
        print(sorted(self.scores_list, reverse=True))
        self.sort_scores_lists()
        print('4444444444: ', len(self.scores_list), self.lowest_qualifing_score)
        print(sorted(self.scores_list, reverse=True))
        return self.scores_list[:self.number_top_scores_to_save], \
               self.score_stds[:self.number_top_scores_to_save], \
               self.train_score_list[:self.number_top_scores_to_save], \
               self.train_stds[:self.number_top_scores_to_save], \
               self.details_list[:self.number_top_scores_to_save]

    def sort_scores_lists(self):
        np_scores_list = np.array(self.scores_list)
        sort_index = np_scores_list.argsort()
        sort_index = np.flip(sort_index)
        # print(sort_index, type(sort_index), sort_index.shape, sort_index.dtype)
        # print(self.scores_list)

        sorted_score = list((np_scores_list[sort_index]))
        print('1: ', sorted_score)
        print(self.train_score_list)
        print(np.array(self.train_score_list))
        print(np.array(self.train_score_list)[sort_index])
        sorted_training_score = np.array(self.train_score_list)[sort_index].tolist()
        # print(type(sorted_score))
        # print('sort: ', sorted_score)
        sorted_std = list(np.array(self.score_stds)[sort_index])
        sorted_training_std = np.array(self.train_stds)[sort_index].tolist()
        # print(sorted_std)
        # print('nn:', self.details_list, type(self.details_list))
        # print(np.array(self.details_list)[sort_index])
        sorted_details = np.array(self.details_list)[sort_index].tolist()
        # print('pppp:', sorted_details, type(sorted_details))
        # print(self.debugger)
        print('999: ', sorted_score)
        self.scores_list = sorted_score[:self.max_iterations]
        print(self.scores_list)
        self.train_score_list = sorted_training_score[:self.max_iterations]
        self.score_stds = sorted_std[:self.max_iterations]
        self.train_stds = sorted_training_std[:self.max_iterations]
        self.details_list = sorted_details[:self.max_iterations]
        # print(self.scores_list)
        print('88: ', self.lowest_qualifing_score)
        self.lowest_qualifing_score = self.scores_list[-1]
        print(self.lowest_qualifing_score)


if __name__ == "__main__":
    print('hello')
    data = pd.read_csv('mango_flouro_rows.csv')
    KVLModelParameterFinder(data)
