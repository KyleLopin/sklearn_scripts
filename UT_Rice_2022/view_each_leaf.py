# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Plot the spectrum for each leave with each day plotted on the graph.  Can view the
graph or save them all to a pdf
"""

__author__ = "Kyle Vitautus Lopin"

# standard libraries
from itertools import cycle

# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# local file
import processing

lines = ["-", "--", "-.", ":"]
line_cycler = cycle(lines)
try:
    fm.fontManager.addfont('THSarabunNew.ttf')
    plt.rcParams['font.family'] = 'TH Sarabun New'
    plt.rcParams['xtick.labelsize'] = 20.0
    plt.rcParams['ytick.labelsize'] = 20.0
except:
    pass  # ttf file into installed
COLORS = plt.cm.cool(np.linspace(0, 1, 18))
ALPHA = 0.9
DATASET = "2"
SET = "second"
SENSOR = "AS7262"
TYPE = "reflectance"
DIFF = False
PROCESSING = "550 nm"


def get_x_columns_and_wavelengths(df: pd.DataFrame):
    x_columns = []
    wavelengths = []
    for column in df.columns:
        if 'nm' in column:
            x_columns.append(column)
            wavelengths.append(column.split()[0])
    x_columns = x_columns[1:]
    wavelengths = wavelengths[1:]
    return x_columns, wavelengths


def make_dead_spectrum(axis, _x_columns, _wavelengths):
    # if TYPE == "reflectance":
    #     # dead_df = pd.read_excel(f"ctrl_and_dead_first_{SENSOR}_{TYPE}.xlsx")
    #     # dead_df = dead_df.loc[dead_df["type exp"] == 'dead'].copy()
    #
    # elif TYPE == 'raw':
    #     dead_df = pd.read_excel(f"dead_leaves_raw.xlsx")
    # else:
    #     raise AttributeError(f"TYPE needs to be 'raw' or 'reflectance' not: {TYPE}")
    dead_df = pd.read_excel(f"dead_leaves_{SENSOR}_{TYPE}.xlsx")
    if PROCESSING == "SNV":
        dead_df[x_columns] = processing.snv(dead_df[x_columns])
    elif PROCESSING == "MSC":
        dead_df[x_columns] = processing.msc(dead_df[x_columns])

    if DIFF:
        dead_df[_x_columns] = dead_df[_x_columns].diff()
    mean = dead_df[_x_columns].mean()
    std = dead_df[_x_columns].std()
    axis.plot(_wavelengths, mean.T, color='black', label="dead leaf")
    axis.fill_between(_wavelengths,
                      (mean - std).T,
                      (mean + std).T, color="black",
                      alpha=0.1)


def make_leaf_figure(leaf_number, _dataset, average=False):

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    data = _dataset.loc[_dataset["Leaf number"] == leaf_number]
    x_columns, wavelengths = get_x_columns_and_wavelengths(data)
    axes.set_ylabel("Reflectance", fontsize=15)
    axes.set_xlabel("Wavelengths (nm)", fontsize=15)
    axes.set_title(f"Leaf number: {leaf_number}\n"
                   f"variety: {data['variety'].unique()[0]}"
                   f"condition: {data['type exp'].unique()[0]}", fontsize=18)
    for day in _dataset["day"].unique():
        daily_data = data.loc[data["day"] == day]
        color_index = int(day/2)
        color = COLORS[color_index]
        if average:
            mean = daily_data[x_columns].mean()
            axes.plot(wavelengths, mean.T, label=day,
                      color=color, ls=line_cycler.next())
        else:
            axes.plot(wavelengths, daily_data[x_columns].T, label=day,
                      color=color, ls=next(line_cycler))
    make_dead_spectrum(axes, x_columns, wavelengths)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes.legend(by_label.values(), by_label.keys())
    return fig


if __name__ == "__main__":
    full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
    _filename = f'{SET}_set_{SENSOR}_every_read_{TYPE}_{PROCESSING}.pdf'
    average = True
    if average:
        _filename = f'{SET}_set_{SENSOR}_every_leaf_{TYPE}_{PROCESSING}.pdf'

    pdf_file = PdfPages(_filename)
    leaves = full_dataset['Leaf number'].unique()
    x_columns, wavelengths = get_x_columns_and_wavelengths(full_dataset)
    if PROCESSING == "SNV":
        full_dataset[x_columns] = processing.snv(full_dataset[x_columns])
    elif PROCESSING == "MSC":
        full_dataset[x_columns] = processing.msc(full_dataset[x_columns])
    elif PROCESSING in x_columns:  # this should be a string of the
        full_dataset[x_columns] = processing.norm_to_column(full_dataset[x_columns], PROCESSING)

    for leaf in leaves:
        print(leaf)
        _fig = make_leaf_figure(leaf, full_dataset)
        pdf_file.savefig(_fig)
        plt.close(_fig)
    pdf_file.close()
