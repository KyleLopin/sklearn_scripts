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

# data = data.loc[(data['position'] == 'pos 2')]
# data = data.loc[(data['integration time'] == 3)]
# data = data.groupby('Leaf number', as_index=True).mean()

x_data, _, data = data_getter.get_data('as7263 roseapple')
chloro_data = data.groupby('Leaf number', as_index=True).mean()

accent_column = chloro_data['Total Chlorophyll (ug/ml)'].to_numpy()
accent_column = accent_column / max(accent_column)
print(accent_column/max(accent_column))

alphas = np.linspace(0.1, 1, 10)
colors = np.zeros((chloro_data.shape[0], 4))
colors[:, 0] = 0.2
colors[:, 1] = 0.6
colors[:, 2] = 0.2
colors[:, 3] = accent_column

spectrum_data_columns = []
wavelengths = []
for column in data:
    if 'nm' in column:
        spectrum_data_columns.append(column)
        wavelengths.append(int(column.split(' nm')[0]))

print(wavelengths)

spectrum_data = data[spectrum_data_columns]

print(data)
print(data.columns)

for i, led in enumerate(data['LED'].unique()):
    # print(led)
    led_spectrum = data[data['LED'] == led]
    # print(led_spectrum)
    led_spectrum = led_spectrum.groupby('Leaf number', as_index=True).mean()
    print('=====')
    # print(led_spectrum)

    led_spectrum = led_spectrum[spectrum_data_columns]
    # print(led_spectrum)
    if i % 3 == 0:
        if i != 0:
            fig.savefig("as7263 roseapple raw {0}.png".format(led))
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.2, 4.5),
                                 constrained_layout=True)

    print(wavelengths)
    fig.suptitle("AS7263 on Roseapple Leaves".format(led))
    axes[i%3].plot(wavelengths, led_spectrum.T, lw=.75)
    axes[i%3].set_title('{0}'.format(led))
    axes[i%3].set_ylabel("Sensor Counts")
    if (i%3) == 2:
        axes[i % 3].set_xlabel("Wavelength (nm)")
    # axes[i % 3].set_xticks([400, 600, 800, 1000])


    # axes[0].annotate("A", xy=(.04, 0.80), xycoords='axes fraction',
    #                  size=24)
    [axes[i%3].lines[j].set_color(color) for j, color in enumerate(colors)]
    #
    #
    # snv_data = processing.snv(led_spectrum)
    #
    # axes[1].plot(wavelengths, snv_data.T, color=colors)
    # axes[1].set_title('Standard Normal Variate Data')
    # axes[1].annotate("B", xy=(.04, 0.80), xycoords='axes fraction',
    #                  size=24)
    # [axes[1].lines[i].set_color(color) for i, color in enumerate(colors)]
    #
    #
    # msc_data, _ = processing.msc(led_spectrum)
    # axes[2].plot(wavelengths, msc_data.T, color=colors)
    # axes[2].set_title("Multiplicative Scatter Correction Data")
    # axes[2].set_xlabel("Wavelength (nm)")
    # axes[2].annotate("C", xy=(.04, 0.80), xycoords='axes fraction',
    #                  size=24)
    # [axes[2].lines[i].set_color(color) for i, color in enumerate(colors)]
    # plt.show()


