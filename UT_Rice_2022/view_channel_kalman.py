# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Plot each of the channels of the sensor and save the figure into a pdf
"""

__author__ = "Kyle Vitatus Lopin"

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
DIFF = False
PROCESSING = "570 nm"  # can be 'SNV', 'MSC', or a spectrum channel
AVERAGE = []
ABSORBANCE = True  # use absorbance or reflectance with False
# can be 'Kalman', 'Trend', or 'Data' to plot lines between the data points
FIT_LINE = "Kalman"


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


def diff_between_varieties(_df: pd.DataFrame, channel: str) -> pd.DataFrame:
    for variety in _df["variety"].unique():
        print(variety)
        _df_slice = _df.loc[_df["variety"] == variety]
        ctrl = _df_slice.loc[_df_slice["type exp"] == 'control'].mean()
        exp = _df_slice.loc[_df_slice["type exp"] == 'งดน้ำ'].mean()
        diff = ctrl[channel] - exp[channel]
        print(_df["day"])
        print(diff)
        plt.plot(_df["day"], diff)


def make_channel_figure(channel, _full_dataset: pd.DataFrame,
                        group_by=[], calc_var_diff=False):
    fig, [pos_axis, vel_axis] = plt.subplots(nrows=2, ncols=1, figsize=(9, 5),
                                             constrained_layout=True)
    group_by.append('day')  # average each day
    _full_dataset = _full_dataset.loc[_full_dataset['variety'] != "กระดาษขาว"]
    # hack, fix this if used again

    fig.suptitle(f"{channel} {SENSOR}, dataset: {SET}", fontsize=TITLE_FONTSIZE)
    if ABSORBANCE:
        _full_dataset[channel] = -np.log10(_full_dataset[channel])
    full_data_backup = _full_dataset.copy()  # kalman filter needs all the data
    if group_by:
        dataset_std = _full_dataset.groupby(group_by, as_index=False
                                            ).std(numeric_only=True)
        # print(dataset_std)
        _full_dataset = _full_dataset.groupby(group_by, as_index=False
                                              ).mean(numeric_only=True)
    for color, variety in zip(COLORS, _full_dataset['variety'].unique()):
        if calc_var_diff:
            data_slice = _full_dataset.loc[_full_dataset["variety"] == variety]
            ctrl = data_slice.loc[data_slice["type exp"] == "control"].groupby('day', as_index=False).mean(numeric_only=True)
            dry = data_slice.loc[data_slice["type exp"] == 'งดน้ำ'].groupby('day', as_index=False).mean(numeric_only=True)
            diff = (ctrl[channel] - dry[channel]) / ctrl[channel]
            pos_axis.scatter(ctrl['day'], diff,
                             color=color, alpha=ALPHA)
            print(variety)
            if FIT_LINE == "Data":
                pos_axis.plot(data_slice['day'], y, ls=ls,
                              color=color, alpha=ALPHA)
            elif FIT_LINE == "Kalman":
                # full_slice = full_data_backup.loc[(full_data_backup["type exp"] == exp_type) &
                #                                   (full_data_backup["variety"] == variety)]
                kalman_fit, kalman_vel = processing.fit_kalman_filter(diff, 'day', channel)
                print(diff)
                print(kalman_fit)
                pos_axis.plot(data_slice['day'].unique(), kalman_fit, ls='solid',
                              color=color, alpha=ALPHA)
                vel_axis.plot(data_slice['day'].unique(), kalman_vel, ls='solid',
                              color=color, alpha=ALPHA, label=variety)

        else:
            for marker, ls, exp_type in zip(['o', 'x'], ['solid', 'dashed'],
                                            ['control', 'งดน้ำ']):
                data_slice = _full_dataset.loc[(_full_dataset["type exp"] == exp_type) &
                                               (_full_dataset["variety"] == variety)]
                y = data_slice[channel]
                pos_axis.scatter(data_slice['day'], y,
                                 marker=marker, color=color, alpha=ALPHA,
                                 label=f"{variety}, {exp_type}")
                if FIT_LINE == "Data":
                    pos_axis.plot(data_slice['day'], y, ls=ls,
                              color=color, alpha=ALPHA)
                elif FIT_LINE == "Kalman":
                    full_slice = full_data_backup.loc[(full_data_backup["type exp"] == exp_type) &
                                                      (full_data_backup["variety"] == variety)]
                    kalman_fit, kalman_vel = processing.fit_kalman_filter(data_slice[channel], 'day', channel)
                    pos_axis.plot(data_slice['day'], kalman_fit, ls=ls,
                                  color=color, alpha=ALPHA)
                    vel_axis.plot(data_slice['day'], kalman_vel, ls=ls,
                                  color=color, alpha=ALPHA)
                if group_by:
                    mean = data_slice[channel]
                    std = dataset_std.loc[(_full_dataset["type exp"] == exp_type) &
                                               (_full_dataset["variety"] == variety)][channel]
                    # axes.fill_between(data_slice['day'],
                    #                   (mean - std).T,
                    #                   (mean + std).T,
                    #                   color=color, alpha=0.05)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.xlabel("Days")
    if ABSORBANCE:
        ylabel = "Absorbance "
    else:
        ylabel = f"{TYPE} "

    if PROCESSING:
        ylabel = f"after normalization at {PROCESSING}"

    pos_axis.set_ylabel(ylabel)
    vel_axis.set_ylabel("Change in absorbance")
    return fig


if __name__ == "__main__":
    full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
    _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}'
    if AVERAGE:
        _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}'
    if ABSORBANCE:
        _filename = f'{SET}_set_{SENSOR}_every_channel_absorbance_{PROCESSING}'

    if DIFF:
        _filename += "_diff"
    _filename += f"_{FIT_LINE}"
    pdf_file = PdfPages(_filename+".pdf")
    leaves = full_dataset['Leaf number'].unique()
    x_columns, wavelengths = get_x_columns_and_wavelengths(full_dataset)

    for channel in x_columns:
        if PROCESSING == "SNV":
            full_dataset[x_columns] = processing.snv(full_dataset[x_columns])
        elif PROCESSING == "MSC":
            full_dataset[x_columns] = processing.msc(full_dataset[x_columns])
        elif PROCESSING in x_columns:  # this should be a string of the
            full_dataset[x_columns] = processing.norm_to_column(full_dataset[x_columns], PROCESSING)

        print(channel)
        _fig = make_channel_figure(channel, full_dataset,
                                   group_by=['variety', "type exp"],
                                   calc_var_diff=DIFF)
        pdf_file.savefig(_fig)
    pdf_file.close()
