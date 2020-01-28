# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

# standard libraries
import pickle
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

__author__ = "Kyle Vitatus Lopin"

# with open('as7265x_2_var_best_inv_ch_total.pickle', 'rb') as pkl:
#     # pickle.dump([r2s, best_details], pkl)
#     r2s, best_details = pickle.load(pkl)
#     # best_details = pickle.load(pkl)
with open('full_data_set_fit.pickle', 'rb') as pkl:
    data = pickle.load(pkl)


print(data)
