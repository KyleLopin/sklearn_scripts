# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import numpy as np
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
    data = pd.read_csv(filename, index_col="Leaf number")
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
        print(data.columns)
        led_key = "LED current"
        if led_key not in data.columns:
            led_key = "led current"
        led_current = format_input_to_list(led_current)
        data = data[data[led_key].isin(led_current)]
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


def get_data_as726x_serial(specimen: str, sensor: str, led_currents: list=["25 mA"],
             int_times: list=[150], positions: list=[2],
             average: bool=False, return_type: str="XY") -> pd.DataFrame:
    """

    :param specimen:
    :param sensor:
    :param led_currents:
    :param int_times:
    :param positions:
    :param average:
    :param return_type:
    :return:
    """
    filename = f"{specimen}_{sensor}_data.csv"
    data = pd.read_csv(filename, index_col="Leaf number")
    print(data.columns)
    print(data.to_string())
    print(data.shape)
    data_columns = []
    y_columns = []
    for column in data.columns:
        if "nm" in column:
            data_columns.append(column)
        elif "Chloro" in column:
            y_columns.append(column)
    print("check")
    y_data = data[y_columns].groupby(["Leaf number"]).mean()
    print(data[y_columns].to_string())
    # print(data.index)
    final_df = pd.DataFrame(index=np.arange(1, 100))
    print(final_df.to_string())

    for led_current in led_currents:
        for int_time in int_times:
            for position in positions:
                new_data_columns = []
                col_prefix = ""
                if len(led_currents) >= 2:
                    col_prefix += f"led current: {led_current} "
                if len(int_times) >= 2:
                    col_prefix += f"int time: {int_time} "
                if len(positions) >= 2:
                    col_prefix += f"pos: {position} "
                for data_column in data_columns:
                    if data_column == "Leaf number":
                        continue
                    new_data_columns.append(col_prefix+f"| {data_column}")

                new_df = data[data["LED current"] == led_current]
                new_df = new_df[new_df["integration time"] == int_time]
                new_df = new_df[new_df["Read number"] == position]
                new_df = new_df[data_columns]
                # print(new_df.shape)
                # print(new_df.columns)
                new_df.columns = new_data_columns
                print('---')
                # print(new_df.to_string())
                print(new_df.shape)
                final_df = pd.concat([final_df, new_df], axis=1)
                print(final_df)
                print('ll', final_df.shape)


    print(new_data_columns)
    print(final_df.to_string())
    print(final_df.shape)
    print(final_df.columns)
    if return_type == "XY":
        return final_df, y_data
    elif return_type == "XYZ":
        return final_df, y_data, data

if __name__ == "__main__":
    get_data_as726x_serial("mango", "as7262", positions=[1, 2, 3])

