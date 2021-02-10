# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, RepeatedKFold

# local files
import data_get
import full_regrs
import processing


sensor = "as7262"

all_regressors = full_regrs.get_all_regrs()
all_transformers = full_regrs.get_transformers()
cv = RepeatedKFold(n_splits=4, n_repeats=15)


for

