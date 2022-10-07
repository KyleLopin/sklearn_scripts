# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Use the coefficient from previously measured rice
leaves to predict chlorophyll levels in the leaves
and model how they change with time in the Undergraduate's project
"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
plt.rcParams.keys()
fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
plt.rcParams['axes.titlesize'] = 20.0
plt.rcParams['figure.titlesize'] = 20.0

TYPE = "raw"
SET = "first"
SENSOR = "AS7262"
MODEL = "pls"

data = pd.read_excel(f"chloro_a_{SENSOR}_{SET}_set_{TYPE}_{MODEL}.xlsx")

x_columns = []
wavelengths = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = data[x_columns]
# chloro_predict = model.predict(x_data)
# print(chloro_predict.shape)
# data["chloro"] = chloro_predict
# data.to_excel("w chloro raw.xlsx", encoding="utf-16")
print(data.columns)
type_exps = data["type exp"].unique()[1:]
type_exps = ['control', 'งดน้ำ']
# type_exps = ['งดน้ำ']
varieties = data["variety"].unique()[1:]
print(varieties)
data = data.groupby(["variety", "type exp", "day"]).mean()
data = data.reset_index()
# print(data.columns)
# leaf_nums = data['Leaf number'].unique()
# leaf_exp_map = {}
# leaf_var_map = {}
# for leaf_num in leaf_nums:
#     # print(data[data['Leaf number'] == leaf_num])
#     df_slice = data[data['Leaf number'] == leaf_num]
#     leaf_exp_map[leaf_num] = df_slice["type exp"].unique().tolist()[0]
#     leaf_var_map[leaf_num] = df_slice["variety"].unique().tolist()[0]
# data = data.groupby(["Leaf number", "day"]).mean()
# data = data.reset_index()
# data["type exp"] = data["Leaf number"].map(leaf_exp_map)
# data["variety"] = data["Leaf number"].map(leaf_var_map)
print(data)
ls_map = {"control": 'solid', "งดน้ำ": 'dashed'}
marker_map = {"control": 'x', "งดน้ำ": 'o'}
colors = ["navy", "turquoise", "darkorange", "magenta"]
data.to_excel("summary_08-18.xlsx", encoding="utf-16")
print(data.index)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
                         constrained_layout=True)
axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
i = 0
for color, name in zip(colors, varieties):
    ser1 = data["variety"] == name
    for exp in type_exps:
        print(f"condition: {name}, {exp}")
        print(f"i = {i}, len axes: {len(axes)}")
        ls = ls_map[exp]
        ser2 = data["type exp"] == exp
        mask = ser1 & ser2
        marker = marker_map[exp]
        axes[i].set_title(f"{name}")
        axes[i].scatter(data.loc[mask, "day"], data.loc[mask, "chloro"],
                        label=f"{name}, {exp}",
                        color=color, marker=marker)
        # plt.scatter(data["day"], data["chloro"],
        #             label=f"{name}, {exp}",
        #             color=color)
    i += 1

    plt.legend()
fig.suptitle(f"{SENSOR} for {SET} set modelled chlorophyll a")
plt.show()

