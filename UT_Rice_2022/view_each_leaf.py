# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Plot the spectrum for each leave with each day plotted on the graph.  Can view the
graph or save them all to a pdf
"""

__author__ = "Kyle Vitautus Lopin"

# installed libraries
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# local files
import get_data

# fm.fontManager.addfont('THSarabunNew.ttf')
# plt.rcParams['font.family'] = 'TH Sarabun New'
# plt.rcParams['xtick.labelsize'] = 20.0
# plt.rcParams['ytick.labelsize'] = 20.0
COLORS = ["navy", "turquoise", "darkorange", "magenta"]
ALPHA = 0.9
SET = "first"
SENSOR = "AS7265x"
TYPE = "reflectance"

FULL_DATASET = pd.read_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
x_columns = []
wavelengths = []
for column in FULL_DATASET.columns:
    if 'nm' in column:
        x_columns.append(column)
        wavelengths.append(column.split()[0])

def make_leaf_figure(leaf_number):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    data = FULL_DATASET.loc[FULL_DATASET["Leaf number"] == leaf_number]
    print(data)

    axes.set_ylabel("Reflectance", fontsize=15)
    axes.set_xlabel("Wavelengths (nm)", fontsize=15)
    for day in FULL_DATASET["day"].unique():
        print(day)
        daily_data = data.loc[data["day"] == day]
        print(daily_data)
        axes.set_title(f"Leaf number: {leaf_number}", fontsize=18)
        mean = daily_data[x_columns].mean()
        axes.plot(wavelengths, mean.T, label=day)
        axes.legend()

    return fig


if __name__ == "__main__":
    _fig = make_leaf_figure(2)
    plt.show()
