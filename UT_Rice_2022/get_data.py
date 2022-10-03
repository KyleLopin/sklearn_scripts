# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
get data sets for the Undergraduate thesis students in August 2022 project of
looking at spectrum of rice leaves grown in pots, some of which will undergo
drought stress
"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
from datetime import datetime
import os

# installed libraries
import pandas as pd
pd.set_option('display.max_columns', None)


def get_data(date, sensor="AS7265x", data_type='reflectance',
             file_end="UT_rice", diff=False, _set=1):
    if sensor not in ["AS7265x", "AS7262"]:
        raise ValueError("sensor has to be 'AS7265x', or 'AS7262'")
    x_columns = []
    wavelengths = []
    try:
        _filename = os.path.join(f"Set_{_set}", f"2022-{date}_{sensor}_{file_end}.xlsx")
        print(f"Getting file: {_filename}")
        data = pd.read_excel(_filename)
        data = data.dropna(axis='index', how='all')
        data = data.fillna(method="ffill")
        for column in data.columns:
            if 'nm' in column:
                x_columns.append(column)
                wavelengths.append(column.split(' nm')[0])
    except:
        raise ValueError(f"dates has to be a string of format (MM-DD) and the data file in the current directory\n"
                         f"you sent {date}")
    data.rename(columns={"การทดลอง": "type exp", "พันธุ์ข้าว": "variety", "หมายเลขกระถาง": "pot number"}, inplace=True)
    print(data.columns)
    print(data)
    if diff:
        data[x_columns] = data[x_columns].diff(axis=1)
        print(data)

    if data_type == "raw":
        return data, wavelengths
    elif data_type == 'reflectance':
        # get average of the white reference
        reference = data[data["variety"] == "กระดาษขาว"]
        reference = reference.groupby("variety").mean(numeric_only=True)
        for column in x_columns:
            data[column] = data[column]/reference[column].values
        return data, wavelengths
    else:
        raise ValueError("data_type has to be 'reflectance' or 'raw'")


def get_all_data(return_type='raw', _set=1, sensor="AS7625x", diff=False):
    if _set == 1:
        dates = ["08-06", "08-08", "08-10", "08-12",
                 "08-14", "08-16", "08-18", "08-20",
                 "08-22", "08-24", "08-26", "08-28",
                 "08-30", "09-01", "09-05", "09-07",
                 "09-09", "09-11"]
        # dates = ["08-06", "08-08", "08-10", "08-12",
        #          "08-14", "08-16", "08-18", "08-20",
        #          "08-22", "08-24", "08-26", "09-01",
        #          "09-07", "09-11"]
    elif _set == 2:
        dates = ["08-27", "08-29", "08-31",
                 "09-02", "09-04", "09-06", "09-08",
                 "09-10", "09-12", "09-14", "09-16",
                 "09-18"]
    final_df = pd.DataFrame()
    data_format = "%m-%d"
    first_day = datetime.strptime(dates[0], data_format)
    for i, date in enumerate(dates):
        new_df, _ = get_data(date, data_type=return_type,
                             sensor=sensor, diff=diff, _set=_set)
        days_from_start = (datetime.strptime(date, data_format) - first_day)
        new_df['day'] = days_from_start
        final_df = pd.concat([final_df, new_df])
    return final_df


if __name__ == "__main__":
    # reflectance = get_data("08-06")
    # print(reflectance)
    SENSOR = "AS7265x"
    TYPE = 'raw'
    SET = "second"
    DIFF = False

    if SET == "first":
        _set = 1
    elif SET == "second":
        _set = 2

    filename = f"{SET}_set_{SENSOR}_{TYPE}"
    if DIFF:
        filename += "_diff"
    df = get_all_data(return_type=TYPE, _set=_set,
                      sensor=SENSOR, diff=DIFF)
    df.to_excel(filename+".xlsx")

    # df.to_excel("second_set_raw.xlsx", encoding="utf-16")
    # df1 = pd.DataFrame({'a': [10], 'b': [20], 'c': [30]})
    # df2 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    # print(df1)
    # print(df1.div(df2, axis='columns'))
    # df, _ = get_data("09-25", file_end="dead", data_type='raw')
    # print(df)
    # df.to_excel("dead_leaves_raw.xlsx")
