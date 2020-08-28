# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
# local files
import data_get

est = LassoCV()
cv = ShuffleSplit(n_splits=100)
# single read
for read in [1, 2, 3]:
    X, Y = data_get.get_data("as7262 mango", integration_time=200, current="25 mA",
                             position=read, return_type="XY")

