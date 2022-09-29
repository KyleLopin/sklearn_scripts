# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
AXIS_LABEL_FONTSIZE = 20
TITLE_LABEL_FONTSIZE = 30
SENSOR = "AS7265x"
TYPE = 'reflectance'
SET = "first"

data = pd.read_excel("first_set_raw.xlsx")
data = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
# data = data[data['type exp'] != "control"]
print(data)
x_columns = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = data[x_columns]
regr = PLSRegression(n_components=6)
# regr = SVR()
# regr = SVR(kernel="poly", degree=2)
# pls.fit(data[x_columns], data["day"])
# y_predict = pls.predict(data[x_columns])
VARIETIES = ['กข43', 'กข79', 'กข85', 'ปทุมธานี 1']
COLORS = {'กข43': "navy", 'กข79': "turquoise",
          'กข85': "darkorange" , 'ปทุมธานี 1': "magenta"}
CONDITIONS = ['control', 'งดน้ำ']
LINESTYLES = {'control': "solid", 'งดน้ำ': "dashed"}
MARKERS = {'control': "o", 'งดน้ำ': "x"}
ctrl_coeffs = {}
dry_coeffs = {}
fig, _axes1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
                           constrained_layout=True)
fig2, _axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
                            constrained_layout=True)
axes = [_axes1.flatten(), _axes2.flatten()]
fig.suptitle(f"{CONDITIONS[0]}", fontsize=TITLE_LABEL_FONTSIZE)
fig2.suptitle(f"{CONDITIONS[1]}", fontsize=TITLE_LABEL_FONTSIZE)
for i, condition in enumerate(CONDITIONS):
    for j, variety in enumerate(VARIETIES):
        # print(data.loc[(data["variety"] == variety)].shape)
        # print(data.loc[data["type exp"] == condition].shape)
        df = data.loc[(data["variety"] == variety) &
                      (data["type exp"] == condition)]
        # df_slice = -np.log(df[x_columns])
        df_slice = df[x_columns]
        regr.fit(df_slice, df["day"])
        y_predict = regr.predict(df_slice)
        print(condition, variety)
        avg_df = pd.concat([pd.DataFrame(y_predict),
                            df['day']], axis=1,
                           ignore_index=True)
        axes[i][j].scatter(df["day"], y_predict,
                           color=COLORS[variety],
                           label=condition,
                           ls=LINESTYLES[condition],
                           marker=MARKERS[condition])
        axes[i][j].set_xlabel("Actual day",
                              fontsize=AXIS_LABEL_FONTSIZE)
        axes[i][j].set_ylabel("Predicted day",
                              fontsize=AXIS_LABEL_FONTSIZE)
        axes[i][j].set_title(f"Variety: {variety}",
                             fontsize=TITLE_LABEL_FONTSIZE)
        axes[i][j].legend()
        if condition == 'control':
            # ctrl_coeffs[variety] = regr.coef_.append(regr.y_mean_)
            ctrl_coeffs[variety] = np.append(regr.coef_, regr.intercept_)
        elif condition == "งดน้ำ":
            # dry_coeefs[variety] = regr.coef_.append(regr.y_mean_)
            dry_coeffs[variety] = np.append(regr.coef_, regr.intercept_)

plt.figure(3)
x = x_columns
x.append("y mean")
BAR_WIDTH = 0.22
plt.title("Control Coefficents")
for j, variety in enumerate(VARIETIES):
    x_ticks = [r + j*BAR_WIDTH for r in range(len(x))]
    print(len(x_ticks), len(ctrl_coeffs[variety]))
    plt.bar(x_ticks, ctrl_coeffs[variety],
            color=COLORS[variety],
            width=BAR_WIDTH,
            label=variety)
    plt.xticks(x_ticks, x, rotation=45)
    plt.legend()

plt.figure(4)
plt.title("Dry Coefficents")
for j, variety in enumerate(VARIETIES):
    x_ticks = [r + j*BAR_WIDTH for r in range(len(x))]
    print(len(x_ticks), len(dry_coeffs[variety]))
    plt.bar(x_ticks, dry_coeffs[variety],
            color=COLORS[variety],
            width=BAR_WIDTH,
            label=variety)
    plt.xticks(x_ticks, x, rotation=45)
    plt.legend()

plt.figure(5)
plt.title("Difference of Coefficents")
for j, variety in enumerate(VARIETIES):
    x_ticks = [r + j*BAR_WIDTH for r in range(len(x))]
    print(len(x_ticks), len(dry_coeffs[variety]))
    plt.bar(x_ticks, ctrl_coeffs[variety]-dry_coeffs[variety],
            color=COLORS[variety],
            width=BAR_WIDTH,
            label=variety)
    plt.xticks(x_ticks, x, rotation=45)
    plt.legend()
# y = data["day"]
# plt.scatter(y, y_predict)
# print(pls.score(data[x_columns], data["day"]))
plt.show()

