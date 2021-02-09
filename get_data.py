# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import pandas as pd


def format_input_to_list(x):
    """
    return the input as a list if it is a string or pass a list through
    :param x: (str or list) input to convert
    :return: (list) list of string(s)
    """
    return x if isinstance(x, list) else [x]


def get_data(specimen: str, sensor: str, led_current: str=None,
             led=None, int_time: int=None, position=None,
             average: bool=False, return_type: str="XY") -> pd.DataFrame:
    """

    :param specimen: (str)
    :param sensor: (str)
    :param led_current: (str)
    :param int_time: (int)
    :param average: (bool)
    :param return_type: (str)
    :return:
    """
    filename = f"{specimen}_{sensor}_data.csv"
    data = pd.read_csv(filename)
    data_columns = []
    y_columns = []
    for column in data.columns:
        if "nm" in column:
            data_columns.append(column)
        elif "Chloro" in column:
            y_columns.append(column)
    print(data_columns)
    print('1:', data.shape)
    print(data.columns)

    if led_current:
        led_current = format_input_to_list(led_current)
        data = data[data['LED current'].isin(led_current)]
        # print(data)
        # data.to_csv("foobar")

    if int_time:
        int_time = format_input_to_list(int_time)
        print(int_time)
        data = data[data["integration time"].isin(int_time)]
    print('2:', data.shape)
    if position:
        position = format_input_to_list(position)
        data = data[data["Read number"].isin(position)]
    print('2b:', data.shape)
    if led:
        led = format_input_to_list(led)
        print(data["led"].unique())
        data = data[data["led"].isin(led)]
    print('3:', data.shape)
    if return_type == "XY":
        return data[data_columns], data[y_columns]
    elif return_type == "XYZ":
        return data[data_columns], data[y_columns], data
