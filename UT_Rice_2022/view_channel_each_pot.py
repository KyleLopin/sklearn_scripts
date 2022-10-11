# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Plot how an individual sensor channel changes for different pots in the UT
students work
"""

__author__ = "Kyle Vitautus Lopin"

# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# local file
import processing

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 15.0
plt.rcParams['ytick.labelsize'] = 15.0
plt.rcParams['axes.labelsize'] = 18.0
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 12

COLORS = {'กข43': "navy", 'กข79': "turquoise",
          'กข85': "darkorange" , 'ปทุมธานี 1': "magenta"}
ALPHA = 0.8
DATASET = 2
if DATASET == 1:
    SET = "first"
elif DATASET == 2:
    SET = "second"
SENSOR = "AS7262"
TYPE = "reflectance"
PROCESSING = None  # can be 'SNV', 'MSC', or a spectrum channel
ABSORBANCE = True  # use absorbance or reflectance with False
FIT_LINE = "Data"
NORM = "กข43"
CHANNEL = "450 nm"


def remove_lows(df:pd.DataFrame, threshold, columns_to_check):
    for column in columns_to_check:
        df = df[df[column] > threshold]
    return df


def get_x_columns_and_wavelengths(df: pd.DataFrame):
    _x_columns = []
    _wavelengths = []
    for column in df.columns:
        if 'nm' in column:
            _x_columns.append(column)
            _wavelengths.append(column.split()[0])
    # _x_columns = _x_columns[1:]
    # _wavelengths = _wavelengths[1:]
    return _x_columns, _wavelengths


def add_mean(_full_dataset, mean_data):
    print("add mean:   ")
    print(_full_dataset)
    print('++++++')
    print(mean_data)
    for day in mean_data.index.unique():
        print(f"day: {day}")
        print()
        print(mean_data.loc[:, day])
        daily_mean = mean_data.loc[:, day].mean()[CHANNEL]
        print(daily_mean)
        if day in _full_dataset.index.unique():
            print('lllll')
            print(_full_dataset.loc[day, CHANNEL])
            _full_dataset.loc[day, CHANNEL] -= daily_mean
    print(_full_dataset)
    # drop days in full dataset not in daily_mean
    for day in _full_dataset.index.unique():
        if day not in mean_data.index.unique():
            _full_dataset = _full_dataset.drop(day)
    return _full_dataset.index, _full_dataset[CHANNEL]


def make_pot_data(channel: str, _full_dataset: pd.DataFrame,
                  pot_number: int, group_by=[]):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    if group_by:
        group_by.append('day')  # average each day
    # hack, fix this if used again
    _full_dataset = remove_lows(_full_dataset, 0.02, x_columns)
    _full_dataset = _full_dataset.loc[_full_dataset['variety'] != "กระดาษขาว"]
    _full_dataset = _full_dataset.loc[_full_dataset['pot number'] == pot_number]
    fig.suptitle(f"{channel} {SENSOR}, dataset: {SET}\n"
                 f"channel: {channel}, pot number: {pot_number}", fontsize=TITLE_FONTSIZE)
    if ABSORBANCE:
        _full_dataset[channel] = -np.log10(_full_dataset[channel])

    mean_df = _full_dataset.groupby(["variety", "day"],
                                    as_index=False).mean(numeric_only=True)
    if group_by:
        _full_dataset = _full_dataset.groupby(group_by, as_index=False
                                              ).mean(numeric_only=True)

    if NORM:
        norm_mean = mean_df.loc[mean_df["variety"] == NORM]
        norm_mean.set_index("day", inplace=True)
        norm_mean = norm_mean[channel]
    for color, variety in zip(COLORS, _full_dataset['variety'].unique()):
        data_slice = _full_dataset.loc[_full_dataset["variety"] == variety]
        data_slice.set_index("day", inplace=True)
        y = data_slice[channel]
        x = data_slice.index
        if NORM:
            x, y = add_mean(data_slice, norm_mean)
        axes.scatter(x, y,
                     color=COLORS[variety], alpha=ALPHA,
                     label=f"{variety}")

        if FIT_LINE == "Kalman":
            kalman_df = mean_df.loc[mean_df['variety'] == variety]  # each pot only has 1 type of exp
            kalman_fit, kalman_vel = processing.fit_kalman_filter(kalman_df[channel])
            axes.plot(kalman_df['day'], kalman_fit,
                      color=color, alpha=ALPHA)

        elif FIT_LINE == "Data":
            x = x.unique()
            y = y.groupby('day').mean()
            # slice_df = mean_df.loc[mean_df['variety'] == variety]  # each pot only has 1 type of exp
            # axes.plot(slice_df['day'], slice_df[channel],
            #           color=color, alpha=ALPHA)
            axes.plot(x, y,
                      color=COLORS[variety], alpha=ALPHA)
    plt.legend()
    return fig


def norm_to_variety(_full_dataset: pd.DataFrame, channel: str):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    print(_full_dataset)
    _full_dataset = remove_lows(_full_dataset, 0.02, x_columns)
    if ABSORBANCE:
        _full_dataset[channel] = -np.log10(_full_dataset[channel])
    data_slice = _full_dataset[['day', "pot number", 'variety', 'type exp', channel]]
    data_slice = data_slice.loc[data_slice['variety'] != 'กระดาษขาว']
    data_slice = data_slice.loc[data_slice['pot number'] != 1]
    data_slice = data_slice.loc[data_slice['pot number'] != np.NaN]

    mean_df = data_slice.groupby(["variety", "day", "pot number"],
                                 as_index=False).mean(numeric_only=True)
    mean_df.set_index(["variety", "pot number", "day"], inplace=True)
    norm_mean = mean_df.loc[NORM]
    print('=======')
    print(norm_mean)
    for pot_num in data_slice["pot number"].unique():
        pot_slice = data_slice.loc[data_slice['pot number'] == pot_num]
        for variety in data_slice['variety'].unique():
            pot_var_slice = pot_slice.loc[pot_slice["variety"] == variety]
            # does not need variety because there is only 1 now
            pot_var_slice.set_index(["pot number", "day"], inplace=True)
            # print(f"pot var slice: {pot_var_slice}")
            # norm mean is indexed with pot number and day
            # this has to be subtraced from the pot_var_slice
            pot_var_slice -= norm_mean.loc[pot_num]
            days = []
            for _pot_num, _day in pot_var_slice.index.values:
                days.append(_day)
            # print(f"pot var slice: {pot_var_slice}")
            axes.scatter(days, pot_var_slice[CHANNEL],
                         color=COLORS[variety], alpha=ALPHA)
    if FIT_LINE == "Data":
        for variety in data_slice['variety'].unique():
            print('oooooo')
            print(mean_df)
            print('=====')
            print(mean_df.loc[variety, :, :])

            norm_mean_df = mean_df.loc[variety, :, :].groupby(level=1).mean()
            print(norm_mean_df)
            print('mix')
            print(norm_mean.loc[:, :])
            # mean_df.loc[variety, :, :] -= mean_df.loc[NORM, :, :].values
            # var_norm = mean_df.loc[variety, :, :].sub(norm_mean.loc[:, :])
            var_norm = norm_mean_df - norm_mean
            print('kkkk')
            print(var_norm)
            print('mmmm')
            mean_var_norm = var_norm.groupby(level=1).mean()
            print(mean_var_norm)
            # x = x.unique()
            # y = y.groupby(['day']).mean()
            days = []
            for _, _day in var_norm.index.values:
                days.append(_day)
            print(f"days: {days}")
            print(var_norm.index.values)
            # axes.plot(days, var_norm, color=COLORS[variety],
            #           alpha=ALPHA)
            axes.plot(mean_var_norm.index, mean_var_norm, color=COLORS[variety],
                      alpha=ALPHA, label=f'{variety}')
    plt.legend(fontsize=16)
    plt.xlabel("Days")
    plt.ylabel("Changes in absorbance coefficents")
    plt.title(f"Changes in absorbance normalized to {NORM} at {CHANNEL}", fontsize=20)
    plt.show()


if __name__ == "__main__":
    full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
    _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}'
    if ABSORBANCE:
        _filename = f'{SET}_set_{SENSOR}_every_channel_absorbance_{PROCESSING}'
    x_columns, wavelengths = get_x_columns_and_wavelengths(full_dataset)
    print(full_dataset["pot number"].unique())
    _filename = f'{SET}_set_{SENSOR}_{CHANNEL}_normed_{NORM}_{TYPE}_{PROCESSING}'

    # pdf_file = PdfPages(_filename + ".pdf")
    # for pot_num in full_dataset["pot number"].unique():
    #     if np.isnan(pot_num):
    #         continue
    #     _fig = make_pot_data(CHANNEL, full_dataset, pot_num,
    #                          group_by=[])
    #     pdf_file.savefig(_fig)
    # pdf_file.close()
    norm_to_variety(full_dataset, CHANNEL)
