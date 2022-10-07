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

COLORS = ["navy", "turquoise", "darkorange", "magenta"]
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
FIT_LINE = None
NORM = "กข43"


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


def make_pot_data(channel, _full_dataset: pd.DataFrame,
                  pot_number: int, group_by=[]):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
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
        norm_mean = 1 - mean_df.loc[mean_df["variety"] == NORM][channel]
        print(norm_mean)
        _full_dataset[channel] += norm_mean
    for color, variety in zip(COLORS, _full_dataset['variety'].unique()):
        data_slice = _full_dataset.loc[_full_dataset["variety"] == variety]
        y = data_slice[channel]
        axes.scatter(data_slice['day'], y,
                     color=color, alpha=ALPHA,
                     label=f"{variety}")

        if FIT_LINE == "Kalman":
            kalman_df = mean_df.loc[mean_df['variety'] == variety]  # each pot only has 1 type of exp
            kalman_fit, kalman_vel = processing.fit_kalman_filter(kalman_df[channel])
            axes.plot(kalman_df['day'], kalman_fit,
                      color=color, alpha=ALPHA)

        elif FIT_LINE == "Data":
            slice_df = mean_df.loc[mean_df['variety'] == variety]  # each pot only has 1 type of exp
            axes.plot(slice_df['day'], slice_df[channel],
                      color=color, alpha=ALPHA)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
    _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}'
    if ABSORBANCE:
        _filename = f'{SET}_set_{SENSOR}_every_channel_absorbance_{PROCESSING}'
    x_columns, wavelengths = get_x_columns_and_wavelengths(full_dataset)
    print(full_dataset["pot number"].unique())
    for pot_num in full_dataset["pot number"].unique():
        print(type(pot_num))
        if pot_num is np.nan:
            continue
        make_pot_data("450 nm", full_dataset, pot_num,
                      group_by=['variety'])
