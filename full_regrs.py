# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.compose import TransformedTargetRegressor
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import isotonic
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, KernelCenterer, Normalizer,
                                   PolynomialFeatures,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)


# speed up the linear SVC
linear_svc = make_pipeline(StandardScaler(), svm.SVR(kernel='linear'))


def invert(x):
    return 1/x


def get_all_regrs():
    regrs = {"Linear regression": linear_model.LinearRegression(),
             # "Perceptron": linear_model.Perceptron(),
             "Lars": linear_model.Lars(),
             "Lasso": linear_model.LassoCV(max_iter=5000),
             # "Passive Aggressive": linear_model.PassiveAggressiveRegressor(),
             "PLS": PLS(n_components=3),
             "Random Forest": ensemble.RandomForestRegressor(),
             "Gradient Boost": ensemble.GradientBoostingRegressor(),
             "Extra Trees": ensemble.ExtraTreesRegressor(max_depth=2),
             "Ada Boost": ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=2),
                                                     n_estimators=250),
             "Gaussian Process": gaussian_process.GaussianProcessRegressor(),
             # "Isotonic": isotonic.IsotonicRegression(),
             "Kernel Ridge": kernel_ridge.KernelRidge(),
             "Ridge CV": linear_model.RidgeCV(),
             # "Exp tranform": TransformedTargetRegressor(regressor=PLS(n_components=3),
             #                                            func=np.exp,
             #                                            inverse_func=np.log),
             # "Log tranform": TransformedTargetRegressor(regressor=PLS(n_components=3),
             #                                            func=np.log,
             #                                            inverse_func=np.exp),
             # "Inv tranform": TransformedTargetRegressor(regressor=PLS(n_components=3),
             #                                            func=invert,
             #                                            inverse_func=invert),
             # "Log regressor": linear_model.LogisticRegressionCV(),
             "ML Perceptron": neural_network.MLPRegressor(),
             "Linear SVR": linear_svc,
             "RBF SVR": svm.SVR(kernel='rbf'),
             "Poly SVR": svm.SVR(kernel='poly'),
             # "Sigmoid SVR": svm.SVR(kernel='sigmoid'),
             "Bayesian Ridge": linear_model.BayesianRidge(),
             "Huber": linear_model.HuberRegressor(),
             # "Poisson": linear_model.PoissonRegressor(),
             "K-neighbors": neighbors.KNeighborsRegressor()}
             # "Radius Neighbors": neighbors.RadiusNeighborsRegressor()}
    return regrs


def get_transformers():
    transforms = {"Standard": StandardScaler(),
                  "Robust": RobustScaler(),
                  "Polynomial": PolynomialFeatures(),
                  "Normalizer": Normalizer(),
                  "Quant normal": QuantileTransformer(n_quantiles=10,
                                                      output_distribution='normal'),
                  "Quant normal": QuantileTransformer(n_quantiles=10,
                                                      output_distribution='uniform'),
                  "Power": PowerTransformer(),
                  "K bins 3 uniform": KBinsDiscretizer(n_bins=3, encode='ordinal',
                                                       strategy='uniform'),
                  "K bins 3 kmeans": KBinsDiscretizer(n_bins=3, encode='ordinal',
                                                      strategy='kmeans'),
                  "K bins 5 uniform": KBinsDiscretizer(n_bins=5, encode='ordinal',
                                                       strategy='uniform'),
                  "K bins 5 kmeans": KBinsDiscretizer(n_bins=5, encode='ordinal',
                                                      strategy='kmeans'),
                  "K bins 7 uniform": KBinsDiscretizer(n_bins=7, encode='ordinal',
                                                       strategy='uniform'),
                  "K bins 7 kmeans": KBinsDiscretizer(n_bins=7, encode='ordinal',
                                                      strategy='kmeans')
                  }
    return transforms


# regers with params: Ridge, Lasso, LassoLars, BayesianRidge, HuberRegressor,
# PoissonRegressor, K-Neighbors, SVR, DecisionTreeRegressor, AdaBoost, Kernal Ridge
