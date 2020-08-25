# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import pandas as pd


def get_data(_type: str, integration_time=None,
             led_current=None, read_number=None, led=None, average=True,
             remove_outlier=False, only_read=False, return_type="XYZ"):
    data = pd.DataFrame()
    data_columns = []
    y_columns = []
    if _type == "as7262 mango":
        data = pd.read_csv("as7262_mango_new.csv")
        for column in data.columns:
            if "nm" in column:
                data_columns.append(column)
            if "Chloro" in column or "chloro" in column:
                y_columns.append(column)


    elif _type == "as7263 betal":
        data = pd.read_csv('as7265x_betal_leaves.csv')

        print(data.shape)

        data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED"]
        chloro_columns = []
        x_columns = ["610 nm", "680 nm", "730 nm",
                     "760 nm", "810 nm", "860 nm"]
        x_data = data[x_columns]
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        data_columns.extend(chloro_columns)
        data = data[data_columns]

        chloro_data = data[chloro_columns]

        return x_data, chloro_data, data

    # print(data.columns)
    if read_number:
        data = data.loc[(data['position'] == read_number)]
    if integration_time:
        data = data.loc[(data['integration time'] == integration_time)]
    if led:
        data = data.loc[(data['LED'] == led)]
    if led_current:
        data = data.loc[(data['LED current'] == led_current)]

    if average:
        data = data.groupby('Leaf number', as_index=True).mean()

    if return_type == "XY":
        return data[data_columns], data[y_columns]
    elif return_type == "XYZ":
        return data[data_columns], data[y_columns], data
