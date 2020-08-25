# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.pipeline import make_pipeline
# local files
import data_get


x_data, _y, full_data = data_get.get_data('as7262 mango', average=False)

print(full_data.columns)
currents = full_data['LED current'].unique()
times = full_data['integration time'].unique()
print(currents, times)
print(full_data['saturation check'].unique())
pls = PLSRegression(n_components=6)
for current in currents:
    for time in times:


        scores = cross_validate(regrs, X_tr, y, cv=cv,
                                scoring=('r2', 'neg_mean_absolute_error'),
                                return_train_score=True)


