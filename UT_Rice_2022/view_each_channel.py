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
LEGEND_FONTSIZE = 14

COLORS = ["navy", "turquoise", "darkorange", "magenta"]
ALPHA = 0.8
DATASET = 2
if DATASET == 1:
    SET = "first"
elif DATASET == 2:
    SET = "second"
SENSOR = "AS7265x"
TYPE = "reflectance"
DIFF = False
PROCESSING = "SNV"
AVERAGE = []


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
    print(_full_dataset.columns)
    print(_full_dataset[channel])
    print(f"full data1 size: {_full_dataset.shape}")
    fig.suptitle(f"{channel} {SENSOR}, dataset: {SET}", fontsize=TITLE_FONTSIZE)
    if group_by:
        dataset_std = _full_dataset.groupby(group_by, as_index=False
                                            ).std(numeric_only=True)
        print(dataset_std)
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
            make_trend_line(axes, ctrl['day'],
                            diff, label_start=f"{variety}",
                            color=color)
            # index = np.isfinite(diff)
            # diff = diff[index]
            # z = np.polyfit(ctrl['day'][index], diff, 1)
            # p = np.poly1d(z)
            # if z[1] >= 0:
            #     _label = f"{variety} y={z[0]:.3f}x+{z[1]:.3f}"
            # else:
            #     _label = f"{variety} y={z[0]:.3f}x{z[1]:.3f}"
            # axes.plot(ctrl['day'], p(ctrl['day']),
            #           color=color, label=_label)
        else:
            for marker, ls, exp_type in zip(['o', 'x'], ['solid', 'dashed'],
                                            ['control', 'งดน้ำ']):
                data_slice = _full_dataset.loc[(_full_dataset["type exp"] == exp_type) &
                                               (_full_dataset["variety"] == variety)]
                axes.scatter(data_slice['day'], data_slice[channel],
                             marker=marker, color=color, alpha=ALPHA)
                make_trend_line(axes, data_slice['day'], data_slice[channel],
                                label_start=f"{variety}, {exp_type}",
                                color=color, ls=ls)
                if group_by:
                    mean = data_slice[channel]
                    std = dataset_std.loc[(_full_dataset["type exp"] == exp_type) &
                                               (_full_dataset["variety"] == variety)][channel]
                    axes.fill_between(data_slice['day'],
                                      (mean - std).T,
                                      (mean + std).T,
                                      color=color, alpha=0.05)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.xlabel("Days")
    if PROCESSING:
        plt.ylabel(f"{TYPE} after {PROCESSING}")
    else:
        plt.ylabel(f"{TYPE}")
    return fig


if __name__ == "__main__":
    full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
    _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}'
    if AVERAGE:
        _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}'

    if DIFF:
        _filename += "_diff"
    pdf_file = PdfPages(_filename+".pdf")
    leaves = full_dataset['Leaf number'].unique()
    x_columns, wavelengths = get_x_columns_and_wavelengths(full_dataset)

    for channel in x_columns:
        if PROCESSING == "SNV":
            full_dataset[x_columns] = processing.snv(full_dataset[x_columns])
        elif PROCESSING == "MSC":
            full_dataset[x_columns] = processing.msc(full_dataset[x_columns])
        print(channel)
        _fig = make_channel_figure(channel, full_dataset,
                                   group_by=['variety', "type exp"],
                                   calc_var_diff=DIFF)
        pdf_file.savefig(_fig)
    pdf_file.close()
