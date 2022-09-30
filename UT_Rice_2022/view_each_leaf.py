# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Plot the spectrum for each leave with each day plotted on the graph.  Can view the
graph or save them all to a pdf
"""

__author__ = "Kyle Vitautus Lopin"

# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
# local files
import get_data

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
COLORS = plt.cm.cool(np.linspace(0, 1, 18))
ALPHA = 0.9
SET = "first"
SENSOR = "AS7265x"
TYPE = "reflectance"

FULL_DATASET = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
print(FULL_DATASET)

x_columns = []
wavelengths = []
for column in FULL_DATASET.columns:
    if 'nm' in column:
        x_columns.append(column)
        wavelengths.append(column.split()[0])
x_columns = x_columns[1:]
wavelengths = wavelengths[1:]


def make_dead_spectrum(axis):
    DEAD_DF = pd.read_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
    DEAD_DF = DEAD_DF.loc[DEAD_DF["type exp"] == 'dead'].copy()
    mean = DEAD_DF[x_columns].mean()
    std = DEAD_DF[x_columns].std()
    axis.plot(wavelengths, mean.T, color='black', label="dead leaf")
    axis.fill_between(wavelengths,
                      (mean - std).T,
                      (mean + std).T, color="black",
                      alpha=0.1)


def make_leaf_figure(leaf_number):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    data = FULL_DATASET.loc[FULL_DATASET["Leaf number"] == leaf_number]
    axes.set_ylabel("Reflectance", fontsize=15)
    axes.set_xlabel("Wavelengths (nm)", fontsize=15)
    axes.set_title(f"Leaf number: {leaf_number}\n"
                   f"variety: {data['variety'].unique()[0]}"
                   f"condition: {data['type exp'].unique()[0]}", fontsize=18)
    for day in FULL_DATASET["day"].unique():
        daily_data = data.loc[data["day"] == day]
        mean = daily_data[x_columns].mean()
        try:
            color_index = int(day/2)
            color = COLORS[color_index]
            # axes.plot(wavelengths, mean.T, label=day,
            #           color=color)
            axes.plot(wavelengths, daily_data[x_columns].T, label=day,
                      color=color)
        except:
            pass  # the dead leaves
    make_dead_spectrum(axes)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes.legend(by_label.values(), by_label.keys())
    return fig


if __name__ == "__main__":
    leaves = FULL_DATASET['Leaf number'].unique()
    DATASET = "1"

    print(leaves)
    pdf_file = PdfPages(f'{SET}_set_{SENSOR}_every_leaf_every_read.pdf')
    for leaf in leaves:
        print(leaf)
        _fig = make_leaf_figure(leaf)
        pdf_file.savefig(_fig)
        plt.close(_fig)
    pdf_file.close()
