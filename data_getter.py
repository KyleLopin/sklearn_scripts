# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import pandas as pd


def get_data(leaf_type: str, remove_outlier=False, only_pos2=False):
    if leaf_type == 'mango':
        data = pd.read_csv('as7262_mango.csv')
        if remove_outlier:
            print(data)
            # data = data.drop(["Leaf: 49"])
            # print(data[data["Leaf number"] == "Leaf: 49"])
            data = data[data["Leaf number"] != "Leaf: 54"]
            data = data[data["Leaf number"] != "Leaf: 41"]
            # data = data[data["Leaf number"] != "Leaf: 37"]
            # data = data[data["Leaf number"] != "Leaf: 38"]
            # data = data[data["Leaf number"] != "Leaf: 33"]
            # data = data[data["Leaf number"] != "Leaf: 34"]
            # data = data[data["Leaf number"] != "Leaf: 35"]
        print(data)

    elif leaf_type == 'roseapple':
        fitting_data = pd.read_csv('as7262_roseapple.csv')



    if only_pos2:
        data = data.loc[(data['position'] == 'pos 2')]
    data = data.groupby('Leaf number', as_index=True).mean()

    data_columns = []
    for column in data.columns:
        if 'nm' in column:
            data_columns.append(column)
    x_data = data[data_columns]

    return x_data, data
