# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# local files
import data_getter
import processing


plt.style.use('seaborn')

# data = pd.read_csv("as7262_mango.csv")
# x_data, data = data_getter.get_data('as7262 roseapple')
x_data, _, data = data_getter.get_data('as7265x ylang')
print(data)
# data = data.loc[(data['position'] == 'pos 2')]
# data = data.loc[(data['integration time'] == 3)]
# data = data.groupby('Leaf number', as_index=True).mean()

# accent_column = data['Total Chlorophyll (ug/ml)'].to_numpy()
accent_column = np.ones_like(data.iloc[:, 1])
# accent_column = accent_column / max(accent_column)
# print(accent_column/max(accent_column))

alphas = np.linspace(0.1, 1, 10)
colors = np.zeros((data.shape[0], 4))
colors[:, 0] = 0.2
colors[:, 1] = 0.6
colors[:, 2] = 0.2
colors[:, 3] = accent_column

spectrum_data_columns = []
wavelengths = []
for column in data:
    if 'nm' in column:
        spectrum_data_columns.append(column)
        wavelengths.append(column.split(' nm')[0])

# wavelengths_nm = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm',
#                '560 nm', '585 nm', '610 nm', '645 nm',
#                '680 nm', '705 nm', '730 nm', '760 nm',
#                '810 nm', '860 nm', '900 nm', '940 nm']

# spectrum_data = data[wavelengths]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
print(spectrum_data_columns)
print(data)
spectrum_data = data[spectrum_data_columns]
print(spectrum_data_columns)
# wavelengths = ['410', '435', '460', '485', '510', '535',
#                '560', '585', '610', '645', '680', '705', '730', '760',
#                '810', '860', '900', '940']
# spectrum_data = data[wavelengths_nm]
#
# print(spectrum_data)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9),
                         constrained_layout=True)
fig.suptitle("AS7263 measurements of Cananga")
print(spectrum_data.shape)
print(data.columns)
# axes[0].plot(wavelengths, spectrum_data.T)
axes[0].plot(spectrum_data.T)
# axes[0].set_ylim([0, 200])
axes[0].set_title('Raw Data')
axes[0].set_ylabel("Raw Sensor Counts")
axes[0].annotate("A", xy=(.04, 0.80), xycoords='axes fraction',
                 size=24)
[axes[0].lines[i].set_color(color) for i, color in enumerate(colors)]


snv_data = processing.snv(spectrum_data)

axes[1].plot(wavelengths, snv_data.T, color=colors)
axes[1].set_title('Standard Normal Variate Data')
axes[1].annotate("B", xy=(.04, 0.80), xycoords='axes fraction',
                 size=24)
[axes[1].lines[i].set_color(color) for i, color in enumerate(colors)]


msc_data, _ = processing.msc(spectrum_data)
axes[2].plot(wavelengths, msc_data.T, color=colors)
axes[2].set_title("Multiplicative Scatter Correction Data")
axes[2].set_xlabel("Wavelength (nm)")
axes[2].annotate("C", xy=(.04, 0.80), xycoords='axes fraction',
                 size=24)
[axes[2].lines[i].set_color(color) for i, color in enumerate(colors)]
plt.show()

