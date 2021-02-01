# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import numpy as np

# local files
import get_data

BACKGROUND = [687.65, 9453.7, 23218.35, 9845.05, 15496.7,
              18118.55, 7023.8, 7834.1, 28505.9, 4040.9,
              5182.3, 1282.55, 2098.85, 1176.1, 994.45,
              496.45, 377.55, 389.75]

plt.style.use('dark_background')


# first add spectrum to model
fig = plt.figure(figsize=(11, 9))
fig.suptitle("Convert sensor data to physical properties",
             size=20)

gs = GridSpec(4, 3, figure=fig)

ax1, ax2, ax3, ax4 = 0, 0, 0, 0
ax3 = fig.add_subplot(gs[0, 2])
for key, spine in ax3.spines.items():
    spine.set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

def make_axis():
    global ax1, ax2, ax3, ax4
    ax1 = fig.add_subplot(gs[:2, :2], zorder=3)
    ax1.set_title("Spectrum data", size=20, y=.85)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[1:, 2])
    ax2.set_title("Chlorophyll levels", size=16)
    ax2.set_xlabel("Leaf number")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Chlorophyll (µg/cm2)")

    ax3 = fig.add_subplot(gs[0, 2])
    for key, spine in ax3.spines.items():
        spine.set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax4 = fig.add_subplot(gs[2:, :2], zorder=1)
    for key, spine in ax4.spines.items():
        spine.set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    ax1.set_ylabel("Fraction of reflectance", size=12)
    ax1.set_xlabel("Wavelength", size=12)

    plt.tight_layout()

x, y = get_data.get_data("mango", "as7265x", int_time=150,
                         position=2, led="b'White'",
                         led_current="25 mA")
print(y.columns)
y = y['Avg Total Chlorophyll (µg/cm2)']

print(x)


def runner(i):
    j = i - 20
    print('i = ', i, j)


    if j == 0:
        make_axis()

    if j == -10:
        ax3.text(0.2, 0.2, "Record spectrum from\n"
                           "many different products\n"
                           "and actual chemical levels",
                 fontsize=15)

    if 0 <= j <= 99:
        # print(x.iloc[i])
        ax1.plot(x.iloc[j]/BACKGROUND, color='limegreen')
        if j == 0:
            ax1.set_xticklabels(x.columns, rotation=60)
        ax2.scatter(j, y.iloc[j], color='limegreen')

    if j == 20:
        con1 = ConnectionPatch(xyA=(.9, .6), xyB=(.2, .7),
                               arrowstyle="simple",
                               coordsA="axes fraction",
                               coordsB="axes fraction",
                               axesA=ax1, axesB=ax2,
                               shrinkB=1, shrinkA=1,
                               fc='navy', lw=4,
                               edgecolor='lightskyblue',
                               mutation_scale=50,
                               zorder=10)
        con1.set_in_layout(False)
        ax1.add_artist(con1)

    if j == 50:
        con2 = ConnectionPatch(xyA=(.5, .1), xyB=(.5, .65),
                               arrowstyle="simple",
                               coordsA="axes fraction",
                               coordsB="axes fraction",
                               axesA=ax1, axesB=ax4,
                               mutation_scale=40,
                               lw=4)
        con2.set_in_layout(False)
        ax1.add_artist(con2)
        props = dict(boxstyle="round",
                     facecolor='wheat', )
        ax4.text(0.1, 0.4, "Machine Learning (ML)\nStatistical Learning methods",
                 bbox=props, fontsize=25,
                 color='black')
    if j == 70:
        con3 = ConnectionPatch(xyA=(.9, .5), xyB=(.2, .4),
                               arrowstyle="simple",
                               coordsA="axes fraction",
                               coordsB="axes fraction",
                               axesA=ax4, axesB=ax2,
                               mutation_scale=40,
                               lw=4)
        con3.set_in_layout(False)
        ax1.add_artist(con3)
        fig.savefig("final1.png")



time = range(120)
ani = FuncAnimation(fig, runner, blit=False,
                    frames=time, repeat=False)
# ani.save('Spectrum_ML_init2.gif', writer='imagemagick')
plt.show()


