# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Script to look through file with spectrum data with different integration times
and see if the integration time affects the reflectance measurement
"""

__author__ = "Kyle Vitatus Lopin"

import pandas as pd


data = pd.read_csv('mango_chloro.csv')

print(data)

grouped_data = data.groupby('integration time').mean()

print('======')
print(grouped_data)

grouped_data.to_csv("int_time_summary.csv")
