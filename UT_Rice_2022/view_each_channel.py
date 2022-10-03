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

DATASET = "2"
SET = "second"
SENSOR = "AS7265x"
TYPE = "reflectance"
DIFF = False
PROCESSING = None


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


def make_channel_figure(channel, full_dataset):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                             constrained_layout=True)
    print(full_dataset.columns)
    data = full_dataset.loc[:, [channel, "day"]]
    print(data)


if __name__ == "__main__":
    full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
    _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}.pdf'
    average = True
    if average:
        _filename = f'{SET}_set_{SENSOR}_every_channel_{TYPE}_{PROCESSING}.pdf'

    pdf_file = PdfPages(_filename)
    leaves = full_dataset['Leaf number'].unique()
    x_columns, wavelengths = get_x_columns_and_wavelengths(full_dataset)
    if PROCESSING == "SNV":
        full_dataset[x_columns] = processing.snv(full_dataset[x_columns])
    elif PROCESSING == "MSC":
        full_dataset[x_columns] = processing.msc(full_dataset[x_columns])

    for channel in x_columns:
        print(channel)
        make_channel_figure(channel, full_dataset)
