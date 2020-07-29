# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"
from matplotlib.animation import FuncAnimation
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

BULK_COLOR = "darkslategray"
DEGRADE_COLOR = "mediumseagreen"
CO2_COLOR = "darkorange"

def k1(temp, a, b, c):
    return 0.01 * np.exp(0.15 * temp)

def k2(temp, a, b, c):
    return -a * np.exp(-b * temp) + c

def temperatures(time):
    return 8 * np.sin(time/10+5.5) + 1 * np.sin(time/20) + \
           33 + 4 * np.random.rand(len(time))

figure = plt.figure(constrained_layout=True, figsize=(8, 9))
heights = [20, 20, 30, 2]
gs = figure.add_gridspec(4, 2)  #, hspace=0.5)

axis1 = figure.add_subplot(gs[1, 0])

temp = np.linspace(20, 50)

axis1.plot(temp, k1(temp, 1, 0.07, 0))
axis1.set_xlabel(u"Temperature (\u00B0C)")
axis1.set_ylabel(u"Reaction rate (s\u207B\u00B9)", size=8)
axis1.set_title(u"k\u2081")


axis2 = figure.add_subplot(gs[1, 1])
axis2.plot(temp, k2(temp, 50, 0.1, 20))
axis2.set_xlabel(u"Temperature (\u00B0C)")
axis2.set_ylabel(u"Reaction rate (s\u207B\u00B9)", size=8)
axis2.set_title(u"k\u2082")

axis3 = figure.add_subplot(gs[2, :])
axis3.set_axis_off()
axis3.set_title("Degradation Model")
props = dict(boxstyle='round',
             facecolor=BULK_COLOR,
             alpha=0.8)
axis3.text(0.05, 0.35, "\n\nBulk Plastic\n\n", bbox=props)
props['facecolor'] = DEGRADE_COLOR
axis3.text(0.40, 0.35, "\n\nDegraded Plastic\n\n", bbox=props)
props['facecolor'] = CO2_COLOR
axis3.text(0.80, 0.35, u"\n\n     C0\u2082     \n\n", bbox=props)

axis3.annotate("", xy=(0.38, 0.60),  xycoords='axes fraction',
            xytext=(0.18, 0.6), textcoords='axes fraction',
            arrowprops=dict(facecolor='darkmagenta', shrink=0.05))

axis3.annotate(u"k\u2081", xy=(0.26, 0.75),
               xycoords='axes fraction', fontsize=15)

axis3.annotate("", xy=(0.78, 0.60),  xycoords='axes fraction',
            xytext=(0.58, 0.6), textcoords='axes fraction',
            arrowprops=dict(facecolor='darkmagenta', shrink=0.05))

axis3.annotate(u"k\u2082", xy=(0.665, 0.75),
               xycoords='axes fraction', fontsize=15)

# time = np.linspace(1, 46, 200)
# bulk = [70000]
# degrade = [0]
# CO2 = [0]
# print(time)
# k1_ = 10
# k2_ = 7
# time2 = [0]

def diff_step(k1, k2):
    global bulk, degrade, CO2, time2
    degrade_step = k1 * bulk[-1] / 400
    CO2_step = k2 * degrade[-1] / 400
    bulk.append(bulk[-1]-degrade_step)
    degrade.append(degrade[-1]+degrade_step-CO2_step)
    CO2.append(CO2[-1]+CO2_step)
    time2.append(time2[-1]+1)
#
#
# for t in time:
#     print(t)
#     diff_step(k1_, k2_)
#
# axis6 = figure.add_subplot(gs[3:5, :])
# print(len(time), len(bulk))
# axis6.plot(time, bulk[:-1], color=BULK_COLOR,
#                label="Bulk Plastic")
# axis6.plot(time, CO2[:-1], color=CO2_COLOR,
#            label=u"C0\u2082")
# plt.legend()
# axis6.set_xlabel("Time (days)")
# axis6.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
# axis6.set_ylabel("Average Molecular Wt.")
#
# plt.show()
# FIGURE 2
time = np.linspace(1, 200, 200)
axis4 = figure.add_subplot(gs[0, :])
temperature_list = temperatures(time)
axis4.plot(time, temperature_list)
axis4.set_xlim([-5, 205])
axis4.set_title("Environmental Data (simulated)")
axis4.set_ylabel(u"Temperature (\u00B0C)")
axis4.set_xlabel("Time")
axis4.spines['right'].set_visible(False)
axis4.spines['top'].set_visible(False)
axis4.spines['bottom'].set_visible(False)
axis4.set_xticklabels([])
axis4.set_xticks([])
axis4.margins(x=.1)
axis4.set_ylim([22, 46])

axis5 = figure.add_subplot(gs[3, :])
axis5.set_ylabel(u"Products")
axis5.spines['right'].set_visible(False)
axis5.spines['top'].set_visible(False)
axis5.spines['bottom'].set_visible(False)
axis5.set_xticklabels([])
axis5.set_xticks([])
axis5.set_xlabel("Time")
axis5.set_xlim([-5, 205])

bulk = [1]
degrade = [0]
CO2 = [0]
time2 = [0]

print(temperature_list)
print(time)
a_circle = None
k1_circle = None
k2_circle = None
con1 = None
con2 = None
con3 = None
con4 = None
con6 = None
con7 = None
legend = None
ann = None

restart = False


def runner(i):
    global a_circle, k1_circle, \
        k2_circle, con1, con2, con3, con4, \
        legend, ann, con6, con7
    if i == 0:
        a_circle = None
        k1_circle = None
        k2_circle = None
        con1 = None
        con2 = None
        con3 = None
        con4 = None
        con6 = None
        con7 = None
        legend = None
        ann = None

    if a_circle:
        a_circle.remove()
        k1_circle.remove()
        k2_circle.remove()
        con1.remove()
        con2.remove()
    print(i, con3)
    if con3:
        con3.remove()
        con4.remove()
    i = int(i)
    temp = temperature_list[i-1]
    print(i, temp)
    a_circle = Ellipse((i, temp), 6, 3, lw=2,
                        color='red', fill=False,
                        clip_on=False)
    axis4.add_patch(a_circle)

    k1_set = k1(temp, 1, 0.07, 0)
    k2_set = k2(temp, 50, 0.1, 20)
    k1_circle = Ellipse((temp, k1_set), 2, 2.5, lw=1.5,
                        color='red', fill=False,
                        clip_on=False)

    axis1.add_patch(k1_circle)
    # k1_circle = axis1.annotate("X", (temp, k1_set))
    k2_circle = Ellipse((temp, k2_set), 2, 1, lw=1.5,
                        color='red', fill=False,
                        clip_on=False)

    axis2.add_patch(k2_circle)

    con1 = ConnectionPatch(xyA=(i, temp), xyB=(temp, k1_set),
                           arrowstyle="-|>", clip_on=False,
                           coordsA="data", coordsB="data",
                           axesA=axis4, axesB=axis1,
                           color="red", shrinkB=8, shrinkA=8.5)
    con1.set_in_layout(False)
    axis4.add_artist(con1)

    con2 = ConnectionPatch(xyA=(i, temp), xyB=(temp, k2_set),
                           arrowstyle="-|>", clip_on=False,
                           coordsA="data", coordsB="data",
                           axesA=axis4, axesB=axis2,
                           color="red", shrinkB=8, shrinkA=8.5)
    con2.set_in_layout(False)
    axis4.add_artist(con2)

    diff_step(k1_set, k2_set)

    axis5.plot(time2, bulk, color=BULK_COLOR,
               label="Bulk Plastic")
    axis5.plot(time2, degrade, color=DEGRADE_COLOR,
               label="Degraded Plastic")
    axis5.plot(time2, CO2, color=CO2_COLOR,
               label=u"C0\u2082")

    if not legend:
        legend = axis5.legend()

    if i == 20:
        annote1 = axis4.text(1, 18, "1) Take environment data\n(temperature, etc)\n" \
                             "and calculate reaction rates", size=12,
                             clip_on=False)

    if i == 50:
        print('=====')
        # annote1.set_text("")

        annote1 = axis1.text(20, 10, "2) Take reaction rates and\n"
                         "plug into decomposition model", size=12,
                     clip_on=False)

        # annote1.set_position((1, 0))
    if i >= 50:
        print('======+++++')
        con3 = ConnectionPatch(xyA=(temp, k1_set), xyB=(0.276, 0.76),
                               arrowstyle="-|>", clip_on=False,
                               coordsA="data", coordsB="axes fraction",
                               axesA=axis1, axesB=axis3, lw=3,
                               color="red", shrinkB=12, shrinkA=8.5)
        con3.set_in_layout(False)
        axis4.add_artist(con3)

        con4 = ConnectionPatch(xyA=(temp, k2_set), xyB=(0.665, 0.75),
                               arrowstyle="-|>", clip_on=False,
                               coordsA="data", coordsB="axes fraction",
                               axesA=axis2, axesB=axis3, lw=3,
                               color="red", shrinkB=12, shrinkA=8.5)
        con4.set_in_layout(False)
        axis2.add_artist(con4)

    if i == 80:
        # annote1.set_text("")

        annote1 = axis3.text(0.3, 0.00, "3) Calculate the amount of\n"
                             "by-product from the model",
                             size=12)

    if i == 90:
        con5 = ConnectionPatch(xyA=(.1, .3), xyB=(2, 1), lw=3,
                               arrowstyle="-|>", clip_on=False,
                               coordsA="data", coordsB="data",
                               axesA=axis3, axesB=axis5,
                               color=BULK_COLOR, shrinkB=8, shrinkA=8.5)
        con5.set_in_layout(False)
        axis5.add_artist(con5)
        ann = axis3.text(0.09, -0.11, "Calculate amount of\nnon-degraded "
                         "plastic", size=10)

    if i == 110:
        con6 = ConnectionPatch(xyA=(.45, .25), xyB=(110, 0.1), lw=3,
                               arrowstyle="-|>", clip_on=False,
                               coordsA="data", coordsB="data",
                               axesA=axis3, axesB=axis5,
                               color=DEGRADE_COLOR, shrinkB=5, shrinkA=8.5)
        con6.set_in_layout(False)
        axis5.add_artist(con6)
        ann = axis5.text(105, 0.8, "Calculate amount of\ndegraded plastic",
                         size=10)
    if i == 130:
        ann.remove()
        con6.remove()
        con7 = ConnectionPatch(xyA=(.85, .25), xyB=(132, 0.6), lw=3,
                               arrowstyle="-|>", clip_on=False,
                               coordsA="data", coordsB="data",
                               axesA=axis3, axesB=axis5, zorder=1,
                               color=CO2_COLOR, shrinkB=5, shrinkA=5)
        con7.set_in_layout(False)
        axis5.add_artist(con7)
        ann = axis3.text(0.67, 0.1,
                         u"Calculate amount of\nCO\u2082 produced",
                         size=10, clip_on=False, zorder=0)
    if i == 150:
        con7.remove()
        ann.remove()
        figure.savefig('PTT final.png')


ani = FuncAnimation(figure, runner, frames=time)
print('===+++ ==== +++ ====')
ani.save('PTTGC_FEM_animation2.gif', writer='imagemagick')
# figure.savefig('PTT final.png')
# plt.show()
