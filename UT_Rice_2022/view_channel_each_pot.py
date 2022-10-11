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
ABSORBANCE = False  # use absorbance or reflectance with False
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
    data_slice = _full_dataset[['day', "pot number", 'variety', 'type exp', channel]]
    data_slice = data_slice.loc[data_slice['variety'] != 'กระดาษขาว']
    data_slice = data_slice.loc[data_slice['pot number'] != 1]
    data_slice = data_slice.loc[data_slice['pot number'] != np.NaN]
    # data_slice.set_index(["variety", "pot number", "day"], inplace=True)
    print(data_slice)
    print('iiiiii')
    mean_df = data_slice.groupby(["variety", "day", "pot number"],
                                 as_index=False).mean(numeric_only=True)
    print(mean_df)
    mean_df.set_index(["variety", "pot number", "day"], inplace=True)
    print(mean_df)
    norm_mean = mean_df.loc[NORM]
    # norm_mean.set_index("day", inplace=True)
    norm_mean = norm_mean
    # normed_df = norm_mean.copy()
    print('=======')
    print(norm_mean)
    for pot_num in data_slice["pot number"].unique():
        pot_slice = data_slice.loc[data_slice['pot number'] == pot_num]
        for variety in data_slice['variety'].unique():
            pot_var_slice = pot_slice.loc[pot_slice["variety"] == variety]
            # does not need variety because there is only 1 now
            pot_var_slice.set_index(["pot number", "day"], inplace=True)
            print(f"pot var slice: {pot_var_slice}")
            # norm mean is indexed with pot number and day
            # this has to be subtraced from the pot_var_slice
            # print(f"indexes: {pot_var_slice.index}")
            # print('=====')
            # print(f"{norm_mean.index}")
            pot_var_slice -= norm_mean.loc[pot_num]
            # print('pppp')
            # print(pot_var_slice.index)
            # print('tttt')
            # print(pot_var_slice.index.values)
            days = []
            for _pot_num, _day in pot_var_slice.index.values:
                days.append(_day)
            # x, y = add_mean(pot_var_slice, norm_mean)
            axes.scatter(days, pot_var_slice[CHANNEL],
                         color=COLORS[variety], alpha=ALPHA,
                         label=f"{variety}")
            # print(f"mean_df: {mean_df}")
            # print(f"index: {mean_df.index}")
            # print(f"pot num: {pot_num}")
            # # pot_var_mean = mean_df.loc[(mean_df["pot number"] == pot_num) &
            # #                            (mean_df["variety"] == NORM)][CHANNEL]
            # pot_var_mean = mean_df.loc[NORM, pot_num, :][CHANNEL]
            # print(f"pot var mean: {pot_var_mean}")
            # print(f"mead df: {mean_df}")
            # print("slice")
            # print(mean_df.index)
            # print(mean_df.loc[variety, pot_num, :])
            # # mean_df = mean_df[(mean_df["pot number"] == pot_num) &
            # #                   (mean_df["variety"] == variety)][CHANNEL] - pot_var_mean
            # mean_df.loc[variety, pot_num, :] -= pot_var_mean
            # print(mean_df)
    if FIT_LINE == "Data":
        for variety in data_slice['variety'].unique():
            var_slice = data_slice.loc[data_slice["variety"] == variety]
            # x, y = add_mean(var_slice, norm_mean)
            print('oooooo')
            print(mean_df)
            # x = x.unique()
            # y = y.groupby(['day']).mean()
            # axes.plot(x, y, color=COLORS[variety],
            #           alpha=ALPHA)
    plt.legend
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
