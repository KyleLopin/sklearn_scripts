# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import pandas as pd


def get_data(leaf_type: str, remove_outlier=False, only_pos2=False):

    if leaf_type == 'as7262 mango':
        print('===')
        data = pd.read_csv('as7262_mango.csv')
        print(data.columns)
        if remove_outlier:
            print(data.columns)
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
        data = data.groupby('Leaf number', as_index=True).mean()
        print(data.columns)
    elif leaf_type == 'as7263 mango':
        data = pd.read_csv('as7265x_mango_leaves.csv')
        print(data.columns)
        # print('=======')
        data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED"]

        chloro_columns = []
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        data_columns.extend(chloro_columns)
        data = data[data_columns]
        # data = data.groupby(['Leaf number', 'LED']).mean()
        print(data)
        print('======')

    elif leaf_type == 'as7265x mango':
        data = pd.read_csv('as7265x_mango_leaves.csv')

    elif leaf_type == 'as7265x mango verbose':
        data = pd.read_csv('as7265x_roseapple_rows.csv')

    elif leaf_type == 'as7262 roseapple':
        data = pd.read_csv('as7262_roseapple.csv')
        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == "as7265x roseaple":
        pass

    else:
        Exception("Wrong input type")

    print(data.columns)
    if only_pos2:
        data = data.loc[(data['position'] == 'pos 2')]


    data_columns = []
    for column in data.columns:
        if 'nm' in column:
            data_columns.append(column)
    x_data = data[data_columns]
    print(data)
    print('++++')
    return x_data, data
