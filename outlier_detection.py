# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import data_getter
import processing

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
x_data, y, data = data_getter.get_data('as7262 betal')
x_data = processing.snv(x_data)
y = y['Total Chlorophyll (ug/ml)']
print(data.index)
results = pd.DataFrame([], index=data.index)
print(results)
outliers_fraction = 0.1
algorithms = [
    ("Elliptic Envelope", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]
plot_num = 1
for name, algorithm in algorithms:
    algorithm.fit(x_data)
    if name == "Local Outlier Factor":
        y_pred = algorithm.fit_predict(x_data)
    else:
        y_pred = algorithm.fit(x_data).predict(x_data)

    print(name)
    print(y_pred)

    plot_num += 1
    results[name] = y_pred

print(results)
results = results.sum(axis=1)
print(results)

results[results <= -2] = -20
results[results > -2] = 20
["Leaf: 35", "Leaf: 46", "Leaf: 5", "Leaf: 54"]  # 100 mA and 50
print(results[results <= -2])
results = results.replace({-20: 'red', 20: 'black'})

# plt.plot(x_data.T, color=results.values)
for i, (index, x) in enumerate(x_data.iterrows()):

    plt.plot(x, color=results.iloc[i])

plt.show()
