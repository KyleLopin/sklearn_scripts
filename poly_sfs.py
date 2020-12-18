# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn import feature_selection
from sklearn.linear_model import LassoCV, SGDRegressor, LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, PolynomialFeatures
# local files
import data_get


plt.style.use('seaborn')

# fitting_data = pd.read_csv('as7262_roseapple.csv')
x_data, _, fitting_data = data_get.get_data('as7262 mango', integration_time=150,
                                            led_current="12.5 mA",
                                            read_number=2)

# regr = PLSRegression(n_components=9)
# regr = LassoCV(max_iter=5000)
regr = LinearRegression()


y = fitting_data['Total Chlorophyll (µg/cm2)']

x_scaled_np = StandardScaler().fit_transform(x_data)
x_scaled_np = PolynomialFeatures(degree=2).fit_transform(x_scaled_np)

print(y)
print(x_scaled_np)

cv = RepeatedKFold(n_splits=5, n_repeats=20)

bins = np.linspace(y.min(), y.max(), 5)
labels = ["1", "2", "3", "4"]
Y_groups = pd.cut(y, bins)

sfs = SFS(regr, floating=True, verbose=2,
          k_features=2, forward=False,
          n_jobs=2,
          scoring='neg_mean_absolute_error', cv=cv)

sfs.fit(x_scaled_np, y)

print("Optimal number of features : %d" % sfs.k_features)
print('Best features :', sfs.k_feature_names_)
print('Best score :', sfs.k_score_)
print(sfs.get_params())
print(sfs)

fig1 = plot_sfs(sfs.get_metric_dict(),
                kind='std_dev',
                figsize=(6, 4))
plt.show()
