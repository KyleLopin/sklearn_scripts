# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data_full = pd.read_csv('mango_flouro_rows.csv')
# data_full = pd.read_csv('mango_chloro_refl3.csv')
# data_full = pd.read_csv('mango_chloro.csv')
data_full = data_full[data_full['position'] == 'pos 2']
# data_full = data_full[data_full['integration time'] == 200]

print(data_full.columns)
y_name = 'Total Chlorophyll (ug/ml)'
# y_name = 'Chlorophyll b (ug/ml)'
y_data = data_full[y_name]

x_data_columns = []
leds = {}
for column in data_full:
    print(column)
    if 'nm' in column:
        x_data_columns.append(column)
        print(column)
        led = column.split(',')[1].strip()
        print(led)
        if led not in leds:
            leds[led] = []
        leds[led].append(column)
print(leds)
pd.options.mode.chained_assignment = 'raise'
plt.style.use('seaborn')

class VariableScan:
    def __init__(self):
        self.test_score = []
        self.test_stdev = []
        self.train_score = []
        self.train_stdev = []

    def add_data(self, test_data, train_data):
        self.test_score.append(test_data.mean())
        self.test_stdev.append(test_data.std())
        self.train_score.append(train_data.mean())
        self.train_stdev.append(train_data.std())

class ModelFit:
    def __init__(self, leds):
        self.leds = leds
        self.scores = object
        self.scores.r2 = VariableScan()
        self.scores.mae = VariableScan()
        self.num_variables = None
        self.columns_used = []
        self.score_matrices = object
        self.score_matrices.r2 = None
        self.score_matrices.mae = None

    def add_scores(self, r2_scores, mae_scores):
        self.scores.r2 = r2_scores
        self.scores.mae = mae_scores


model_fits = []


def pls_component_scanner(x_data, y_data, max_components=None):
    # print(x_data.shape)
    # print('=============')
    # pca = PCA()
    # pca.fit(x_data)
    # print(pca.explained_variance_ratio_)
    # cum_sum = np.cumsum(pca.explained_variance_ratio_)
    # print(cum_sum)
    # print(np.searchsorted(cum_sum, 0.9))
    # print(np.searchsorted(cum_sum, 0.999))
    # print(np.searchsorted(cum_sum, 0.9999))
    # ham
    best_score = -10000
    best_conditions = None
    best_pls_components = 0
    if not max_components:
        max_components = int(x_data.shape[1] / 2)
    print(max_components)
    print(x_data.columns)
    best_scores = []
    for i in range(1, max_components+1):
        # print('running i =', i)
        scores, num_vars, best_columns, top_score = variable_scanner(x_data, y_data, i)
        print('running i =', i, top_score)
        best_scores.append(top_score)
        if top_score > best_score:
            best_score = top_score
            best_conditions = best_columns
            best_pls_components = i
    # plt.plot(best_scores)
    # plt.show()
    plt_x_data = x_data[best_conditions]
    print(plt_x_data)
    print(best_pls_components)

    # plot_pls_results(plt_x_data, y_data, best_pls_components, 1)
    return best_score, best_conditions, best_pls_components


def variable_scanner(x_data, y_data, num_pls_components):
    best_score = -10000
    best_columns = None
    num_variables = []
    r2_data = VariableScan()
    mae_data = VariableScan()
    while x_data.shape[1] >= num_pls_components:
        pls = PLSRegression(num_pls_components)
        # new_model_fit = ModelFit()
        # print(x_data.shape[1])
        # print('number to cut: ', number_to_cut, num_pls_components)
        pls.fit(x_data, y_data)
        # cv_splitter = GroupKFold(n_splits=4)
        cv_splitter = GroupShuffleSplit(n_splits=10,
                                        test_size=0.35)
        group_splitter = data_full['Leaf number']
        local_scores = cross_validate(pls, x_data, y_data,
                                      cv=cv_splitter,
                                      return_train_score=True,
                                      groups=group_splitter,
                                      scoring=['r2', 'neg_mean_absolute_error'])

        # data.scores.r2.add_data(local_scores['test_r2'],
        #                         local_scores['train_r2'])
        # data.scores.mae.add_data(local_scores['test_neg_mean_absolute_error'],
        #                         local_scores['train_neg_mean_absolute_error'])
        # data.num_variables.append(x_data.shape[1])
        #
        # if data.scores.r2.test_score[-1] > best_score:
        #     best_score = data.scores.r2.test_score[-1]
        #     best_columns = x_data.columns
        r2_data.add_data(local_scores['test_r2'],
                          local_scores['train_r2'])
        mae_data.add_data(local_scores['test_neg_mean_absolute_error'],
                          local_scores['train_neg_mean_absolute_error'])


        if mae_data.test_score[-1] > best_score:
            best_score = mae_data.test_score[-1]
            best_columns = x_data.columns

        num_variables.append(x_data.shape[1])
        # to increase the speed cut up to 1% of all the lowest variables
        number_to_cut = int(x_data.shape[1] / 100)
        if number_to_cut == 0:
            number_to_cut = 1
        sorted_coeff = np.argsort(np.abs(pls.coef_[:, 0]))

        columns_to_drop = x_data.columns[sorted_coeff[:number_to_cut]]

        # this causes SettingWithCopyWarning
        # x_data.drop(columns=columns_to_drop, inplace=True)
        x_data = x_data.drop(columns=columns_to_drop)
        # print(best_score, r2_data.test_score[-1], x_data.shape)
    return (r2_data, mae_data), num_variables, best_columns, best_score


def add_led_old(leds_already, columns_already):
    for new_led, new_columns in leds.items():
        total_leds = leds_already + new_led
        new_data = ModelFit(total_leds)

        all_columns = columns_already + new_columns
        print(all_columns)
        x_data = data_full[all_columns]
        # fill in
        variable_scanner(x_data, )


def cut_variable(x_data, y_data, num_pls_components):
    pass


def plot_pls_results(x_data, y_data, pls_components, num_variables):
    pls = PLSRegression(pls_components)

    cv_splitter = GroupShuffleSplit(n_splits=1,
                                    test_size=0.35,
                                    random_state=6)  # 1
    group_splitter = data_full['Leaf number']
    print('111111111')
    print(x_data)


    for train_index, test_index in cv_splitter.split(x_data, y_data, group_splitter):
        # print(train_index, test_index)
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        pls.fit(x_train, y_train)
        y_pred_train = pls.predict(x_train)
        y_pred_test = pls.predict(x_test)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)

        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        print(r2_test, mae_test)
        print(r2_train, mae_train)
        print(r2_score(y_train, y_pred_train))
        print(r2_score(y_test, y_pred_test))
        plt.scatter(y_train, y_pred_train, c='blue', label='Training Set')
        plt.scatter(y_test, y_pred_test, c='red', label='Test Set')

        _line = np.linspace(0.2, 1.2)

        # plt.plot(_line, _line, c='indigo', linestyle='dashed')
        #
        # plt.plot(_line, _line + .06, c='darkslategray', linestyle='dashed')
        # plt.plot(_line, _line - .06, c='darkslategray', linestyle='dashed')
        #
        # left_annote_pos = 0.20
        # plt.annotate("Training Median Absolute Error = {}".format(0.059),
        #              (left_annote_pos, 1.1), fontsize=12)
        # # plt.annotate("Testing Median Absolute Error = {}".format(0.07),
        # #              (left_annote_pos, 1.02), fontsize=12)
        #
        # plt.annotate(u"Training R\u00B2 = {}".format(0.83),
        #              (left_annote_pos, .95), fontsize=12)
        #
        # # plt.annotate(u"Testing R\u00B2 = {}".format(0.82),
        # #              (left_annote_pos, .89), fontsize=12)
        # plt.xlabel('Meausured Chlorophyll b (ug/ml)', fontsize=16)
        # plt.ylabel('Predicted Chlorophyll b (ug/ml)', fontsize=16)
        # plt.title("Chlorophyll b Model for AS7262\nbased on 2-Component\nPartial Least Squared Model",
        #           fontsize=18)
        # plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.scatter(y_pred_train, y_train, c='blue', label='Training Set')
        plt.scatter(y_pred_test, y_test, c='red', label='Test Set')
        plt.show()


def as7265x_scanner(max_pls_components):
    print(leds)
    print(leds['White LED'])
    print(data_full[leds['White LED']])
    # total_leds = leds['White LED'] + leds['UV (405 nm) LED'] + leds['IR LED']
    total_leds = leds['White LED'] + leds['505 nm LED'] + leds['IR LED']
    print(total_leds)
    # total_leds += leds['505 nm LED']
    # total_leds = leds['White LED']
    x = data_full[x_data_columns]
    x = data_full[total_leds]
    print(x)

    y = data_full[y_name]
    # y = 1 / y
    # pls_components = 32
    # scores, num_vars, best_columns, best_score = variable_scanner(x, y, pls_components)

    best_score, best_columns, best_pls_components = pls_component_scanner(x, y, max_pls_components)
    print(best_score)

    print(best_columns)
    print(best_pls_components)
    print(best_columns.shape)
    plt_data = x[best_columns]
    print(plt_data)
    plot_pls_results(plt_data, y_data, best_pls_components, 1)


def as7262_scanner():
    best_score = -10000
    best_columns = None
    best_pls_components = 0
    x_data = data_full[x_data_columns]
    # x_data = 1 / x_data
    # y_data = data_full['Total Chlorophyll (ug/ml)']
    y_data = data_full[y_name]
    y_data = 1 / y_data
    # y_data = np.exp(y_data)
    # y_data = np.log(1 / y_data) # .82 with raw numbers
    print('===========')
    print(x_data)
    pls_component_scanner(x_data, y_data, 4)
    # scores, num_vars, best_columns, best_score = variable_scanner(x_data, y_data, 2)
    # print(scores)
    plt_data = x_data[best_columns]
    print('++++++++++')
    print(plt_data)
    plot_pls_results(plt_data, y_data, 4, 1)


def add_led(previous_led_set):
    included_columns = []
    for led_set in previous_led_set:
        pass


def led_scanner(filename, previous_leds=[], max_pls_components=None):
    """
    Go through a set of leds, for each set of leds go through all the
    leds and add them to the previous set and go through the pls
    variable scanner and find the best fit and save all the data
    to a pickle file with filename
    :param filename: file to save the dictionary of led set as the
    key and the CV score as the value
    :param previous_leds: list of leds to start with
    :param max_pls_components: maximum number of pls components to test
    :return:
    """
    best_score = -10000
    included_columns = []
    best_scores = {}
    # for old_led in previous_leds:
    #     print(old_led)
    #     included_columns += leds[old_led]
    print('=====++++++++++')
    # print(included_columns)
    # print(len(included_columns))
    # print(leds)
    # print(previous_leds)
    # ===========================
    for old_leds in previous_leds:
        print(old_leds, previous_leds)
        included_columns += leds[old_leds]

        # =======================
        print('included: ', included_columns)
    for new_led, new_columns in leds.items():
        print(new_led)
        if new_led in old_leds:
            continue

        # print(new_columns)
        total_leds = new_columns + included_columns
        # print(total_leds)
        # print(len(total_leds))
        x_data = data_full[total_leds]

        score, _, pls_components = pls_component_scanner(x_data, y_data, max_pls_components)

        if score > best_score:
            best_score = score
        print('led = {} | best score = {} | pls components = {}'.format(new_led,
                                                                        score,
                                                                        pls_components))
        best_scores[new_led] = score

    # ===========================
    with open(filename, 'wb') as f:
        pickle.dump(best_scores, f, pickle.HIGHEST_PROTOCOL)

    for led, score in best_scores.items():
        print(led, score)


def show_data(filename):
    """
    Print the scores and led set of the scores from an
    led scanner pickle file
    :param filename: name of led scanner pickle file
    :return:
    """
    best_value = -np.inf
    best_key = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(data)
    for key, value in data.items():
        print(key, value)
        if value > best_value:
            best_value = value
            best_key = key
    print('best set:')
    print(best_key, best_value)


def get_best_sets(filename, num_sets):
    """
    Look through an led scanner pickle file (filename) and return
    the best led, or set of leds
    :param filename: pickle filename with led scanner data
    :param num_sets: maximum number of sets to return
    :return: led or set of leds with the best scores
    """
    values = []
    led_set = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    for key, value in data.items():
        values.append(value)
        led_set.append(key)
    values = np.array(values)
    led_set = np.array(led_set)
    print(values)
    print(led_set)
    sort_indices = np.argsort(values)
    print(sort_indices)
    print(values[sort_indices[::-1]])
    print(led_set[sort_indices[::-1]])
    best_led_sets = led_set[sort_indices[::-1]]
    print(best_led_sets[:num_sets])
    return best_led_sets[:num_sets]


if __name__ == '__main__':
    print('===========')
    # as7265x_scanner(100)
    # x = data_full[x_data_columns]
    # total_leds = leds['White LED']
    # x = data_full[total_leds]
    # y = data_full[y_name]
    # pls_component_scanner(x, y)
    # as7262_scanner()
    # led_scanner(['White LED'])
    # led_scanner(['505 nm LED'], 20)
    # with open('led_scanner_white_525.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # for key, value in data.items():
    #     print(key)
    #     print(value)
    print('=====---=====')
    show_data('led_scanner_white_525.pickle')
    # best_led_set = get_best_sets('led_scanner_2_505.pickle', 20)
    # led_scanner('led_scanner_2.pickle', previous_leds=best_led_set)
    # led_scanner('led_scanner_2_505.pickle', previous_leds=['505 nm LED'])
    # led_scanner('led_scanner_white.pickle', previous_leds=['White LED'])
    # led_scanner('led_scanner_white_525.pickle', previous_leds=['White LED', '525 nm LED'])
