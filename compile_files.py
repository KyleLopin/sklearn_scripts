# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import pandas as pd


new_cols = ['Total Chlorophyll (µg/mg)', 'Chlorophyll a (µg/mg)',
            'Chlorophyll b (µg/mg)', 'Total Chlorophyll (µg/cm2)',
            'Chlorophyll a (µg/cm2)', 'Chlorophyll b (µg/cm2)']


def make_mango_as7262(sensor_data, chloro_data):
    # some chloro b data from new leaves are negative so make them zero
    chloro_data[chloro_data < 0] = 0
    mean_ = chloro_data.groupby('Leaf number', as_index=True).mean()
    std_ = chloro_data.groupby('Leaf number', as_index=True).std()
    # for col in new_cols:
    #     chloro_data['Avg ' + col] = 0
    #     chloro_data['STD ' + col] = 0
    # for i, row in chloro_data.iterrows():
    #     for col in new_cols:
    #         mean_col = col
    #         leaf_num = row['Leaf number']
    #         avg_value = mean_[mean_col][leaf_num]
    #         std_value = std_[mean_col][leaf_num]
    #         chloro_data.loc[i, 'Avg ' + col] = avg_value
    #         chloro_data.loc[i, 'STD ' + col] = std_value
    print(mean_)
    for col in new_cols:
        sensor_data[col] = 0
        sensor_data[col+' std'] = 0

    for i, row in sensor_data.iterrows():
        for col in new_cols:
            leaf_num = row["Leaf number"]
            mean_value = mean_[col][leaf_num]
            std_value = std_[col][leaf_num]

            sensor_data.loc[i, col] = mean_value
            sensor_data.loc[i, col+" std"] = std_value

    print(sensor_data)
    sensor_data.to_csv("as7262_mango_new.csv")


def combine_files(sensor_data_filename, chloro_data_filename, new_filename):
    sensor_data = pd.read_csv(sensor_data_filename)
    chloro_data = pd.read_csv(chloro_data_filename)
    # add columns of chloro data to sensor data
    chloro_columns = []

    for column in chloro_data.columns:
        if "Chloro" in column:
            chloro_columns.append(column)
    print(chloro_columns)
    new_df = sensor_data.copy()
    print(new_df.columns)
    for column in chloro_columns:
        new_df[column] = ""
    print(new_df.columns)
    for i, row in new_df.iterrows():
        new_leaf_no = row["Leaf number"]
        chloro_row = chloro_data.loc[chloro_data["Leaf No."] == new_leaf_no]
        print(i)
        # print(chloro_row)
        for column in chloro_columns:
            # print(row.name, column)
            # print(new_df.loc[row.name])
            new_row = new_df.loc[row.name]
            # print('+++++++', column)
            # print(new_row[column])

            chloro_pts = chloro_row[column]
            # print('=====', chloro_pts.values)
            # print('llll', new_df.columns)
            # print('999', new_df.loc[row.name, column])
            new_df.loc[row.name, column] = chloro_pts.values[0]
    new_df.to_csv(new_filename)


if __name__ == "__main__":
    # _sensor_data = pd.read_csv("as7262 mango n.csv")
    # _chloro_data = pd.read_csv("new_chloro_mango_cur.csv")
    # _chloro_data = _chloro_data.drop(["Spot", "Mean leaf weight (mg)", ], axis=1)
    # make_mango_as7262(_sensor_data, _chloro_data)
    combine_files("Sugarcane AS7265X.csv",
                  "sugarcane_chlorophyll_data.csv",
                  "sugarcane_as7265x_data.csv")
