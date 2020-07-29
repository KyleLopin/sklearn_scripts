# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import pandas as pd


def get_data(_type: str, integration_time=None,
             LED_current=None, position=None, LED=None,
             remove_outlier=False, only_pos2=False):
    if _type == "AS7262 Mango":
        pass

    elif _type == "AS7263 Betal":
        data = pd.read_csv('as7265x_betal_leaves.csv')
        if position:
            data = data.loc[(data['position'] == position)]
        if integration_time:
            data = data.loc[(data['integration time'] == integration_time)]
        if LED:
            data = data.loc[(data['LED'] == LED)]
        if LED_current:
            data = data.loc[(data['LED current'] == LED_current)]
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
