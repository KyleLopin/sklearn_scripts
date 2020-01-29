# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge, Lars, LassoLars
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.linear_model import ElasticNet, ARDRegression, BayesianRidge
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ridge_regression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import LinearSVR, NuSVR, SVR

# local files
import data_getter
import processing

plt.style.use('ggplot')

x_data, data = data_getter.get_data('as7263 mango')
chloro_data = data.groupby('Leaf number', as_index=True).mean()

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)

y_columns = ['Total Chlorophyll (ug/ml)',
             'Chlorophyll a (ug/ml)',
             'Chlorophyll b (ug/ml)']

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

def method_tester(x_data, y_data, modeler, test_size=0.25):
    # calculate full train set
    # y_data = data[y_name]
    modeler.fit(x_data, y_data)
    y_predict = modeler.predict(x_data)
    r2_train = r2_score(y_data, y_predict)
    mae_train = mean_absolute_error(y_data, y_predict)

    # # calculate a test set
    local_scores = cross_validate(modeler, x_data, y=y_data,
                                  cv=cv, scoring=['r2', 'neg_mean_absolute_error'])
    r2_test = local_scores['test_r2'].mean()
    mae_test = -local_scores['test_neg_mean_absolute_error'].mean()
    print(r2_train, mae_train, r2_test, mae_test)
    return r2_train, mae_train, r2_test, mae_test

models_to_test = [PLS(n_components=1), PLS(n_components=2),
                  PLS(n_components=3), PLS(n_components=4),
                  Lasso(alpha=1), Lasso(alpha=0.1),
                  Lasso(alpha=0.01), Lasso(alpha=0.001),
                  LassoLars(alpha=1), LassoLars(alpha=0.1),
                  LassoLars(alpha=0.01), LassoLars(alpha=0.001),
                  Ridge(alpha=0.01, max_iter=5000),
                  Ridge(alpha=0.001, max_iter=5000),
                  Ridge(alpha=0.0001, max_iter=5000),
                  Ridge(alpha=0.00001, max_iter=5000),
                  Lars(), GaussianProcessRegressor(),
                  GradientBoostingRegressor(), SVR(), LinearSVR(), NuSVR(),
                  LogisticRegression(), LinearRegression(), SGDRegressor(),
                  ElasticNet(), ARDRegression(), BayesianRidge(),
                  HuberRegressor(), RANSACRegressor(), TheilSenRegressor(),
                  PassiveAggressiveRegressor(),
                  AdaBoostRegressor(), BaggingRegressor(), GradientBoostingRegressor(),
                  RandomForestRegressor(n_estimators=10, max_depth=2),
                  ExtraTreesRegressor(n_estimators=10, max_depth=2),
                  KernelRidge()]

model_names = ["PLS 1-component", "PLS 2-component",
               "PLS 3-component", "PLS 4-component",
               "Lasso alpha 1", "Lasso alpha 0.1",
               "Lasso alpha 0.01", "Lasso alpha 0.001",
               "LassoLars alpha 1", "LassoLars alpha 0.1",
               "LassoLars alpha 0.01", "LassoLars alpha 0.001",
               "Ridge alpha 0.01", "Ridge alpha 0.001",
               "Ridge alpha 0.0001", "Ridge alpha 0.00001",
               "Lars", "Guassian Regression",
               "Gradient Boosting",
               "SVR", "LinearSVR", "NuSVR",
               "LogisticRegression", "LinearRegression", "SGDRegressor",
               "ElasticNet", "ARDRegression", "BayesianRidge",
               "HuberRegressor", "RANSACRegressor", "TheilSenRegressor",
               "PassiveAggressiveRegressor",
               "AdaBoostRegressor", "BaggingRegressor", "GradientBoostingRegressor",
               "RandomForestRegressor", "ExtraTreesRegressor",
               "Kernel Ridge"
               ]

y_transformations = [None, (np.exp, np.log), (np.log, np.exp), (np.log1p, np.expm1),
                     (np.reciprocal, np.reciprocal),
                     QuantileTransformer(n_quantiles=10),
                     QuantileTransformer(n_quantiles=10, output_distribution='normal'),
                     PowerTransformer()]

x_transformations = [None, np.exp, np.log, np.reciprocal, np.log1p, np.expm1]

transormation_names = ["None", "Exponential", "Logarithm",
                       "Reciprocol",  "Logarithm plus 1", "Exponential minus one",
                       "Quantile Transform", "Normal Quantile Transform",
                       "Power Transform"]

results_total = []
results_a = []
results_b = []

results = [results_total, results_a, results_b]

# print(data)
# print(data[data["LED"] == "White LED"].groupby("Leaf number", as_index=True).mean())


for z, result in enumerate(results):
    for y, led in enumerate(['White LED']):
        for i, test_model in enumerate(models_to_test):
            for j, y_transformation in enumerate(y_transformations):
                for k, x_transformation in enumerate(x_transformations):

                    led_data = data[data["LED"] == led].groupby("Leaf number", as_index=True).mean()
                    x_data = led_data[data_columns]
                    y_data = led_data[y_columns[z]]

                    print(y_columns[z], led)

                    title = "{0}\ny transformer: {1}" \
                            "\nx transformer: {2}".format(model_names[i],
                                                          transormation_names[j],
                                                          transormation_names[k])

                    if x_transformation:
                        x_data_new = x_transformation(x_data)
                    else:
                        x_data_new = x_data.copy()

                    if type(y_transformation) == tuple:
                        _model = TransformedTargetRegressor(regressor=clone(test_model),
                                                            func=y_transformation[0],
                                                            inverse_func=y_transformation[1])
                    elif y_transformation:
                        _model = TransformedTargetRegressor(regressor=clone(test_model),
                                                            transformer=y_transformation)
                    else:
                        _model = clone(test_model)

                    print(title)

                    try:
                        # fit_n_plot_estimator(x_data_new, y_columns,
                        #                      _model, title)
                        r2_train, mae_train, r2_test, mae_test = method_tester(x_data_new,
                                                                               y_data,
                                                                               _model)
                        result.append([model_names[i], transormation_names[j],
                                       transormation_names[k], led, r2_train,
                                       mae_train, r2_test, mae_test])
                        # fit_n_plot_estimator(x_data_new, y_columns,
                        #                      _model, title)
                        # method_tester(x_data_new, y_columns, _model)
                    except Exception as e:
                        print("FAIL===============>>>>>>>>>")
                        print(e)

    results_pd = pd.DataFrame(result, columns=["Model", "y transform", "x transform", "LED", "r2 train", "mae train", "r2 test", "mae test"])
    print(results_pd)

    results_pd.to_csv("as7263 {0} results raw.csv".format(y_columns[z]))
