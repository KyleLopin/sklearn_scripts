# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed files
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.linear_model import RidgeCV, LassoCV, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
# local files
import get_data
import processing


BACKGROUND = [687.65, 9453.7, 23218.35, 9845.05, 15496.7,
              18118.55, 7023.8, 7834.1, 28505.9, 4040.9,
              5182.3, 1282.55, 2098.85, 1176.1, 994.45,
              496.45, 377.55, 389.75]

# BACKGROUND = [800.65, 10453.7, 28218.35, 9845.05, 15496.7,
#               18118.55, 7023.8, 7834.1, 28505.9, 4040.9,
#               5182.3, 1282.55, 2098.85, 1176.1, 994.45,
#               496.45, 377.55, 389.75]
x, y = get_data.get_data("mango", "as7265x", int_time=150,
                             position=2, led="b'White'",
                             led_current="25 mA")
x_columns = x.columns
x = x.iloc[:, 8:]
x.columns = x_columns[:10]
# x = pd.DataFrame(x, columns=x_columns)
print(x.shape)
# ham
y = y['Avg Total Chlorophyll (µg/cm2)']
x_reflect = x / BACKGROUND[8:]
print(x_reflect)
# ham
# x_snv = processing.snv(x_reflect)
# x_msc, _ = processing.msc(x_reflect)
# x_robust = RobustScaler().fit_transform(x_msc)
plt.style.use('dark_background')
x_msc = x_reflect
pls = PLS(n_components=6)

pls.fit(x, y)
x_fit = pls.predict(x)
pls.fit(x_msc, y)
svr = SVR()
svr.fit(x_msc, y)
print(svr.score(x, y))

lasso = LassoCV()
print(y.values)
print(type(y.values))
print(x.values)
print(type(x.values))
lasso.fit(x_reflect, y.values)
print(lasso.score(x_reflect, y))
x_fit = lasso.predict(x_reflect)
# ham

# x_fit = pls.predict(x_msc)

coeff_final = lasso.coef_
print(coeff_final)
print('-======')
# print(pls.x_mean_)
# print(pls.y_mean_)
# print('bbbbb')
# print(ridge.coef_)



def vis_ax(ax, state=False):
    for key, spine in ax.spines.items():
        spine.set_visible(state)
    ax.get_xaxis().set_visible(state)
    ax.get_yaxis().set_visible(state)


def ax_turn_on(ax):
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

# make predict figure
# plt.style.use('seaborn')
# fig = plt.figure(figsize=(5, 5))
# gs = GridSpec(1, 1, figure=fig)
# ax = fig.add_subplot(gs[0, 0])
# # plt.scatter(range(100), y, color='limegreen')
#
# # plt.scatter(range(100), x_fit, color='limegreen', marker='x')
# plt.scatter(y, x_fit)
# vis_ax(ax, False)
# ax_turn_on(ax)
# # ax.set_xlabel("Leaf number", fontsize=12)
# # ax.set_ylabel("Predicted chlorophyll (µg/cm\u00B2)", fontsize=12)
# # ax.set_title("Model predicted chlorophyll", fontsize=18)
# ax.set_xlabel("Actual chlorophyll (µg/cm\u00B2)", fontsize=14)
# ax.set_ylabel("Predicted chlorophyll (µg/cm\u00B2)", fontsize=14)
# ax.set_title("LASSO model fitting", fontsize=18)
#
# print(lasso.score(x_reflect, y))
y_predict = lasso.predict(x_reflect)
print(mean_absolute_error(y, y_predict))
ham
# ax.plot([0, 90], [0, 90], ls='--',
#         color='magenta', lw=2)
#
# plt.show()

fig = plt.figure(figsize=(11, 9), constrained_layout=False)
fig.suptitle("Machine Learning Process\n"
             "Least Absolute Shrinkage and Selection Operator\n"
             "(LASSO)",
             size=20)
plt.subplots_adjust(top=0.8)
gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 2])
ax1 = fig.add_subplot(gs[0, :3], zorder=4)
ax2 = fig.add_subplot(gs[2, :3], zorder=3)
ax3 = fig.add_subplot(gs[2, :2], zorder=2)
ax4 = fig.add_subplot(gs[0, :2], zorder=2)
ax5 = fig.add_subplot(gs[0, 2], zorder=2)
ax6 = fig.add_subplot(gs[2, 2], zorder=2)
ax_table = fig.add_subplot(gs[1, :2], zorder=2)
N_nums = 6
S_SHIFT = 0
LEAF_NUMBER = 0
avg_read = []
diff = []
coeffs = []
results = []
table = None
con = None
diff_line = None
refl_line = None

vis_ax(ax1, False)
vis_ax(ax2, False)
vis_ax(ax3, False)
vis_ax(ax4, False)
vis_ax(ax5, False)
vis_ax(ax6, False)
vis_ax(ax_table, False)
props = dict(boxstyle="round",
             facecolor='wheat', )
text1 = []
text2 = []
arrows = []
conns = []


def make_arrow(x, ax):
    arrow = ax.arrow(x, .8, .1, -.4,
                     transform=ax2.transAxes,
                     shape='full', lw=2,
                     head_width=0.025, head_length=0.1)
    arrows.append(arrow)


def make_arrows(ax, start_x):
    make_arrow(start_x, ax)
    make_arrow(start_x+.05, ax)
    make_arrow(start_x+.1, ax)


def clear_list(_list):
    for element in _list:
        element.set_visible(False)

frame_delay = [0]
delay_times = [3, 8, 5, 4, 4,
               4, 2, 4, 3, 3,
               5, 5, 5, 5]
for dt in delay_times:
    frame_delay.append(frame_delay[-1]+dt)
frame_i = -1
print(frame_delay)

def runner(i):
    global table, con, diff, results, \
        diff_line, refl_line, frame_i

    frame_i += 1
    print('i = ', i, table, frame_i)
    if i == frame_delay[0]:
        # put intro text

        tx1 = ax1.text(0.5, 0.75, "Algorithm to convert spectrum data to\n"
                                  "chemical concentrations",
                       fontsize=15, bbox=props, color='black',
                       ha='center')
        text1.append(tx1)

    if i == frame_delay[1]:
        # clear_list(text1)
        # ax1.plot(x_robust.T, color='limegreen')
        tx1 = ax1.text(0.5, 2.0, "Different techniques can be used\n"
                                 "based on the problem such as\n"
                                 "Partial Least Squared, LASSO\n"
                                 "Support Vector Machines (SVM)",
                       transform=ax2.transAxes,
                       bbox=props,
                       fontsize=15, color='black',
                       ha='center')
        text1.append(tx1)
    if i == frame_delay[2]:
        # ax2.subplot2grid((0, 1), (1, 1))

        # gs = GridSpec(2, 3, figure=fig)
        ax2.clear()
        vis_ax(ax2, False)
        ax3.plot(x_reflect.T, color='limegreen')
        ax3.set_xticklabels(x.columns, rotation=60)
        ax_turn_on(ax3)
        ax3.set_ylim([-.15, .8])
        ax3.set_title("Reflectance data",
                      y=1.2, fontsize=16)
        ax3.set_ylabel("Reflectance")
        # ax3.set_ylim([-0.4, 0.6])
        ax3.set_zorder(5)
    if i == frame_delay[3]:
        clear_list(text1)
        tx1 = ax5.text(0.5, .85, "How does LASSO convert\n"
                                 "spectrum to chemical concentration?",
                       bbox=props,
                       fontsize=12, color='black',
                       ha='center')
        text2.append(tx1)
        ax5.set_zorder(5)

    if i == frame_delay[4]:
        tx1 = ax5.text(0.5, .55, "Take an individual reading",
                       bbox=props,
                       fontsize=12, color='black',
                       ha='center')
        text2.append(tx1)

    if i == frame_delay[5]:
        ax3.cla()
        refl_line, = ax3.plot(x_reflect.iloc[LEAF_NUMBER], color='limegreen',
                              label='individual reflectance')
        # ax3.plot(pls.x_mean_, color='darkorange',
        #          label='mean reflectance')
        # ax3.set_ylim([])
        ax3.set_xticklabels(x.columns, rotation=60)
        ax3.set_ylabel("Reflectance")
        ax3.set_ylim([-.15, .8])
        ax3.legend(loc='upper left')

    # if i == frame_delay[5]:
    #     tx1 = ax5.text(0.5, .65, "The individual reflectance is\n"
    #                              "subtracted from the mean",
    #                    bbox=props,
    #                    fontsize=12, color='black',
    #                    ha='center')
    #     text2.append(tx1)
    #     diff_line, = ax3.plot(x_msc.iloc[LEAF_NUMBER] - pls.x_mean_,
    #                           color='cyan', label='Difference')
    #     ax3.legend()
    #
    if i == frame_delay[6]:
        tx1 = ax5.text(0.5, .15, "LASSO then makes a set\n"
                                "of coefficients from a\n"
                                 "subset of the spectrum",
                       bbox=props,
                       fontsize=12, color='black',
                       ha='center')
        text2.append(tx1)
    #
    # if i == frame_delay[8]:
    #     tx1 = ax5.text(0.5, .05, "Multiply model coefficients\n"
    #                             "by spectrum difference and sum the results",
    #                    bbox=props,
    #                    fontsize=12, color='black',
    #                    ha='center')
    #     text2.append(tx1)

    if i == frame_delay[7]:
        # make model
        # clear_list(text1)
        ax4.set_zorder(5)
        ax_turn_on(ax4)
        print(pls.coef_.T[0]/pls.x_std_)
        print(coeff_final)
        ax4.bar(np.arange(len(lasso.coef_)), coeff_final, color='magenta',
                tick_label=x.columns)

        ax4.set_ylabel("Model coefficients")
        ax4.set_xticklabels(x.columns, rotation=60)
        # ax4.set_ylim([-8.5, 6.5])
        ax4.set_title("PLS model coefficient weights",
                      fontsize=16, y=0.9)

    if i == frame_delay[8]:
        tx1 = ax5.text(0.5, -.2, "The LASSO coefficients are then\n"
                                 "multiplied by the sensor data",
                       bbox=props,
                       fontsize=12, color='black',
                       ha='center')
        text2.append(tx1)

    if i == frame_delay[9]:
        # clear_list(text2)
        print(pls.coef_.T[0])
        print(pls.coef_.T[0].tolist())
        nums = [[], [], []]
        colors = [[], [], []]
        col_colors = []
        # avg_read = []
        row_labels = ["sensor read", "coeffs", "sensor * coeffs"]
        column_labels = []
        row_colors = ["limegreen", "magenta", "white"]

        for n in range(N_nums):
            print(x_msc.iloc[0, n])
            column_labels.append(x.columns[S_SHIFT+n])
            read = x_msc.iloc[LEAF_NUMBER, S_SHIFT+n]
            nums[0].append("{:0.4f}".format(read))
            nums[1].append("")
            nums[2].append("")

            colors[0].append("limegreen")
            colors[1].append("black")
            colors[2].append("black")

            col_colors.append("white")
            avg_pt = pls.x_mean_[S_SHIFT+n]
            # avg_read.append("{:0.4f}".format(avg_pt))
            # diff.append("{:0.4f}".format(read-avg_pt))
            # print('l', pls.coef_.T[0][n])
            coeffs.append("{:0.0f}".format(coeff_final[S_SHIFT+n]))
            # print(pls.y_std_[0]*(read-avg_pt)*coeff_final[S_SHIFT+n])
            results.append("{:0.1f}".format((read*coeff_final[S_SHIFT+n])))

        print('nums')
        print(nums)
        table = ax_table.table(cellText=nums,
                               cellColours=colors,
                               rowLabels=row_labels,
                               colLabels=column_labels,
                               rowColours=row_colors,
                               colColours=col_colors,
                               loc='center')
        table.scale(1, 1.5)
        table_props = table.properties()
        table_cells = table_props['children']
        for cell in table_cells:
            cell.get_text().set_fontsize(12)
            cell.get_text().set_color('black')
        # plt.tight_layout()

        print(type(ax_table))
        ax_table.set_zorder(7)

    # if i == frame_delay[9]:
    #     print('aaa')
    #     print(avg_read)
    #     print(diff)
    #     print(coeffs)
    #     print(results)
    #     # print(table)
    #     for i in range(N_nums):
    #         print(avg_read[i])
    #         table._cells[(2, i)]._text.set_text(avg_read[i])
    #         table._cells[(2, i)].set_facecolor('darkorange')

    # if i == frame_delay[10]:
    #     for i in range(N_nums):
    #         table._cells[(3, i)]._text.set_text(diff[i])
    #         # table._cells[(3, i)].get_text().set_color('black')
    #         print('-----')
    #         print(dir(table._cells[(3, i)]))
    #         print('-----')
    #         table._cells[(3, i)].set_facecolor('cyan')

    if i == frame_delay[10]:
        for j in range(N_nums):
            table._cells[(2, j)]._text.set_text(coeffs[j])
            table._cells[(2, j)].set_facecolor('magenta')

    if i == frame_delay[12]:
        for i in range(N_nums):
            table._cells[(3, i)]._text.set_text(results[i])
            table._cells[(3, i)].set_facecolor('white')

    if i == frame_delay[13]:

        print(x_msc.iloc[0])

        print(x_fit)
        ax6.scatter(1, x_fit[LEAF_NUMBER], marker='x')
        ax_turn_on(ax6)
        ax6.set_zorder(5)
        ax6.set_title("Predicted chlorophyll levels")
        ax6.set_ylabel('Avg Total Chlorophyll (µg/cm\u00B2)')
        ax6.set_xlabel("Leaf number")
        # ax6.axhline(pls.y_mean_, color='magenta', ls='--')
        con = ConnectionPatch(xyA=(1, -0.2), xyB=(1, x_fit[LEAF_NUMBER]),
                              arrowstyle="simple",
                              coordsA="axes fraction",
                              coordsB='data',
                              shrinkB=6,
                              facecolor='lightgreen',
                              axesA=ax_table, axesB=ax6,
                              mutation_scale=40,
                              lw=2, zorder=10)
        con.set_in_layout(False)
        ax_table.add_artist(con)

        tx1 = ax5.text(0.5, -0.6, "\u2211 sensor read*coeffs",
                       bbox=props,
                       fontsize=18, color='black',
                       ha='center')
        text1.append(tx1)
        ax5.set_zorder(12)
        # print(dir(pls))
        # print(pls.x_std_, pls.y_std_)

    if i == frame_delay[14]:
        clear_list(text2)
        tx1 = ax5.text(0.5, -.2, "Each leaf has the\n"
                                 "chlorophyll level predicted\n"
                                 "by the LASSO model",
                       bbox=props,
                       fontsize=16, color='black',
                       ha='center')
        text2.append(tx1)

    if i > frame_delay[14]:
        leaf_no = i - frame_delay[-1] + 1
        con.set_visible(False)

        diff = []
        results = []
        data_pt = []
        # leaf_data = x_msc.iloc[i-53]
        for n in range(N_nums):
            avg_pt = pls.x_mean_[S_SHIFT + n]
            read = x_reflect.iloc[leaf_no, S_SHIFT + n]
            data_pt.append("{:0.4f}".format(read))
            # diff.append("{:0.4f}".format(read - avg_pt))
            results.append("{:0.1f}".format((read - avg_pt) * coeff_final[S_SHIFT + n]))
        for j in range(N_nums):
            table._cells[(1, j)]._text.set_text(data_pt[j])
            # table._cells[(3, j)]._text.set_text(diff[j])
            table._cells[(3, j)]._text.set_text(results[j])
        # print(results)
        # print(read)
        # print(leaf_no, i)
        # read = x_msc.iloc[leaf_no]
        ax6.scatter(leaf_no, x_fit[leaf_no],
                    color='limegreen', marker='x')
        # print('mmmm', read - avg_pt)
        # print(read-pls.x_mean_)
        # print(read)
        # print(pls.x_mean_)
        # diff_line.set_ydata(read - pls.x_mean_)
        refl_line.set_ydata(x_reflect.iloc[leaf_no])
        fig.savefig("final_lasso_explainer.png")

time = range(140)
ani = FuncAnimation(fig, runner, repeat=False,
                    frames=time, interval=1000)
ani.save("LASSO_explanation.gif", writer='imagemagick')
# plt.show()



