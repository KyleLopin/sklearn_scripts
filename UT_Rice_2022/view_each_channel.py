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
PROCESSING = None  # can be 'SNV', 'MSC', or a spectrum channel
AVERAGE = []
ABSORBANCE = True  # use absorbance or reflectance with False
# can be 'Kalman', 'Trend', or 'Data' to plot lines between the data points
FIT_LINE = "Data"


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


def remove_lows(df:pd.DataFrame, threshold, columns_to_check):
    for column in columns_to_check:
        df = df[df[column] > threshold]
    return df


def make_trend_line(_axis, _x, _y, label_start="",
                    color="black", ls='solid'):
    _index = np.isfinite(_x) & np.isfinite(_y)
    _coeffs = np.polyfit(_x[_index], _y[_index], 1)
    _fit = np.poly1d(_coeffs)
    if _coeffs[1] >= 0:
        label_start += f" y={_coeffs[0]:.5f}x+{_coeffs[1]:.5f}"
    else:
        label_start += f" y={_coeffs[0]:.5f}x{_coeffs[1]:.5f}"
    _axis.plot(_x, _fit(_x), color=color,
               label=label_start, ls=ls)


def make_channel_figure(channel, _full_dataset: pd.DataFrame,
                        group_by=[], calc_var_diff=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    group_by.append('day')  # average each day
    _full_dataset = _full_dataset.loc[_full_dataset['variety'] != "กระดาษขาว"]
    # hack, fix this if used again
    _full_dataset = remove_lows(_full_dataset, 0.02, x_columns)
    fig.suptitle(f"{channel} {SENSOR}, dataset: {SET}", fontsize=TITLE_FONTSIZE)
    if ABSORBANCE:
        _full_dataset[channel] = -np.log10(_full_dataset[channel])
    print(_full_dataset)
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
            axes.scatter(ctrl['day'], diff,
                         color=color, alpha=ALPHA)
            # make trend line
            if FIT_LINE == "Trend":
                make_trend_line(axes, ctrl['day'],
                                diff, label_start=f"{variety}",
                                color=color)
        else:
            for marker, ls, exp_type in zip(['o', 'x'], ['solid', 'dashed'],
                                            ['control', 'งดน้ำ']):
                data_slice = _full_dataset.loc[(_full_dataset["type exp"] == exp_type) &
                                               (_full_dataset["variety"] == variety)]
                y = data_slice[channel]
                axes.scatter(data_slice['day'], y,
                             marker=marker, color=color, alpha=ALPHA,
                             label=f"{variety}, {exp_type}")
                if FIT_LINE == "Trend":
                    make_trend_line(axes, data_slice['day'], y,
                                    label_start=f"{variety}, {exp_type}",
                                    color=color, ls=ls)
                elif FIT_LINE == "Data":
                    axes.plot(data_slice['day'], y, ls=ls,
                              color=color, alpha=ALPHA)
                elif FIT_LINE == "Kalman":
                    print('[[', variety, exp_type)
                    full_slice = full_data_backup.loc[(full_data_backup["type exp"] == exp_type) &
                                                      (full_data_backup["variety"] == variety)]
                    kalman_fit, kalman_vel = processing.fit_kalman_filter(data_slice[channel], 'day', channel)
                    axes.plot(data_slice['day'], kalman_fit, ls=ls,
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
        ylabel = f"Absorbance"
    else:
        ylabel = f"{TYPE}"
    if PROCESSING:
        ylabel += f" after {PROCESSING}"

    plt.ylabel(ylabel)
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
        plt.show()
        pdf_file.savefig(_fig)
    pdf_file.close()
