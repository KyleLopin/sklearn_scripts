# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
# local files
import helper_functions as funcs

as7265x_files = funcs.get_all_files_with_stub('as7265x', 'betal')
print(as7265x_files)

all_data = pd.DataFrame()
starting = True

# for file in as7265x_files:
#     file_data = pd.read_csv(file)
#     if starting:
#         all_data = file_data
#         starting = False
#     else:
#         print('appending', all_data.shape)
#         all_data = all_data.append(file_data)
for file in as7265x_files:
    print(file)
    file_data = pd.read_csv(file)
    print(file_data.columns)
    all_data = pd.concat([all_data, file_data], sort=False)
    # print(all_data)

print(all_data)

all_data.to_csv('foobar6.csv')
