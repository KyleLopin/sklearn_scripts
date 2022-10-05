# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SET = "first"
SENSOR = "AS7262"
TYPE = "reflectance"

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
AXIS_LABEL_FONTSIZE = 20
TITLE_LABEL_FONTSIZE = 30

VARIETIES = ['กข43', 'กข79', 'กข85', 'ปทุมธานี 1']
COLORS = {'กข43': "navy", 'กข79': "turquoise",
          'กข85': "darkorange" , 'ปทุมธานี 1': "magenta"}
CONDITIONS = ['control', 'งดน้ำ']
LINESTYLES = {'control': "solid", 'งดน้ำ': "dashed"}
MARKERS = {'control': "o", 'งดน้ำ': "x"}

# df = pd.read_excel(f"modeled_health_{SET}_{SENSOR}_reflectance.xlsx")
df = pd.read_excel(f"daily_modeled_health_{SET}_{SENSOR}_{TYPE}.xlsx")
df = df.loc[df['day'] != 30].copy()
df_summary = df.groupby(["type exp", "variety", "day"],
                        as_index=False).mean(numeric_only=True)
df_std = df.groupby(["type exp", "variety", "day", "pot number"]
                    ).std(numeric_only=True)
fitted_curves = {}


def sigmoid(x, L, x0, k, b):
    return L / (1+np.exp(-k*(x-x0))) + b


for i, condition in enumerate(CONDITIONS):
    plt.figure(i)
    plt.title(f"{condition}, {SENSOR}, {SET}",
              fontsize=TITLE_LABEL_FONTSIZE)
    for j, variety in enumerate(VARIETIES):
        df_cond = df_summary.loc[(df_summary["variety"] == variety) &
                                 (df_summary["type exp"] == condition)]
        print(condition, variety)
        plt.scatter(df_cond["day"], df_cond["modeled_health"],
                    color=COLORS[variety], label=variety,
                    ls=LINESTYLES[condition])
        # make fitted line
        p0 = [min(df_cond["modeled_health"])-1, 10, 1, 1]
        popt, pcov = curve_fit(sigmoid, df_cond["day"],
                               df_cond["modeled_health"], p0,
                               method='dogbox', maxfev=50000)
        print(popt)
        fit = sigmoid(df_cond["day"], *popt)
        plt.plot(df_cond['day'], fit)
    plt.figure(i)
    plt.xlabel("Days", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Modeled health", fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend()

plt.show()