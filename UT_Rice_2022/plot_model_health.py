# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import pandas as pd

VARIETIES = ['กข43', 'กข79', 'กข85', 'ปทุมธานี 1']
COLORS = {'กข43': "navy", 'กข79': "turquoise",
          'กข85': "darkorange" , 'ปทุมธานี 1': "magenta"}
CONDITIONS = ['control', 'งดน้ำ']
LINESTYLES = {'control': "solid", 'งดน้ำ': "dashed"}
MARKERS = {'control': "o", 'งดน้ำ': "x"}

df = pd.read_excel("modeled_health.xlsx")
df_summary = df.groupby(["type exp", "variety", "day"], as_index=False).mean()
df_std = df.groupby(["type exp", "variety", "day"]).std()
print(df_summary)
print(df_summary.columns)
# fig, _axes1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
#                            constrained_layout=True)
# fig2, _axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8),
#                             constrained_layout=True)
# axes = [_axes1.flatten(), _axes2.flatten()]
for i, condition in enumerate(CONDITIONS):
    plt.figure(i)
    plt.title(CONDITIONS)
    for j, variety in enumerate(VARIETIES):
        df_cond = df_summary.loc[(df_summary["variety"] == variety) &
                                 (df_summary["type exp"] == condition)]
        print(condition, variety)
        plt.scatter(df_cond["day"], df_cond["modeled_health"],
                    color=COLORS[variety], label=condition,
                    ls=LINESTYLES[condition])
    plt.legend()

plt.show()