# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import os
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# local files
import helper_functions as funcs

# for root, directory, files in os.walk(os.getcwd()):
#     print(root, directory, files)
#     for filename in files:
#         print(root+'/'+filename)

background = [250271, 176275, 216334, 219763, 230788, 129603]

# as7262_data_file = "betal_as7262_data.csv"
as7262_data_file = 'as7265x_betal_full.csv'
# chloro_data_file = r"/Users/kylesmac/PycharmProjects/plant_chloro_data_analysis/betal data/Wildbetal leafbush OD.xlsx"
chloro_data_file = "betal_chloro.csv"
as7262_data = pd.read_csv(as7262_data_file)
chloro_data = pd.read_csv(chloro_data_file)
# chloro_data = pd.read_excel(chloro_data_file, sheet_name='summary')
as7262_data.set_index('Leaf number', inplace=True)
chloro_data.set_index('Leaf number', inplace=True)

print(as7262_data)

print('==========')
print(chloro_data)
print(as7262_data.columns)
# fix leaf number in chlorophyll data file
# kk = []
# for i in range(1, 61):
#     for j in range(5):
#         kk.append('Leaf: {0}'.format(i))
#
# kk = pd.DataFrame(kk)
# print(kk)
# kk.to_csv('foobar.csv')
# make the top leaf / middle leaf / bottom leaf column
# kk = []
# for i in range(1, 61):
#     for j in ['Top leaf', 'Middle leaf', 'Bottom leaf']:
#         for k in range(10):
#             kk.append(j)
#
# kk = pd.DataFrame(kk)
# print(kk)
# kk.to_csv('foobar3.csv')

chloro_mean = chloro_data.groupby(by="Leaf number").mean()
chloro_std = chloro_data.groupby(by="Leaf number").std()
print(chloro_mean)
# chloro_mean.to_csv('foobar.csv')
# chloro_std.to_csv('foobar2.csv')
full_data = pd.merge(as7262_data, chloro_data, right_index=True, left_index=True)
print(full_data)
full_data.to_csv('foobar7.csv')
