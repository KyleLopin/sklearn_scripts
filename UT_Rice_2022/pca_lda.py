# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


DIMENSIONS = 2
CUMULATIVE = False
ADD_DEAD_LEAVES = True
LDA_COLUMN = "type exp"

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0

pd.set_option('display.max_columns', None)
# data = pd.read_excel("2022-08-06_AS7265x_UT_rice.xlsx", keep_default_na=False)
# data = pd.read_excel("2022-08-10_AS7265x_UT_rice.xlsx")
# data, wavelenghts = get_data.get_data("08-14")
# data = pd.read_excel("w chloro raw first set.xlsx")
data = pd.read_excel("first_set_reflectance.xlsx")
SET = "first"
SENSOR = "AS7265x"
TYPE = "reflectance"

if ADD_DEAD_LEAVES:
    dead_leaf_data = pd.read_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
    dead_leaf_data = dead_leaf_data.loc[dead_leaf_data["health"] == 0]
    data = pd.concat([data, dead_leaf_data])
    data['health'] = data['health'].fillna(1.0)
data = data.loc[data["variety"] != 'กระดาษขาว']
print(data)
# data = pd.read_excel("first_4_days.xlsx")
# data = data.dropna(axis='index', how='all')
# data = data.fillna(method="ffill")

x_columns = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
# data = data[data["type exp"] !='control']
marker_map = {'X': "control", 'o': "งดน้ำ"}
marker_map = {"control": 'x', "งดน้ำ": 'o'}
LINESTYLES = {"control": 'dashed', "งดน้ำ": 'solid'}
type_exps = ['control', 'งดน้ำ']
print('======')
print(type_exps)
# the reference thai word needs to be removed first
names = data["variety"].unique()

pca = PCA(n_components=DIMENSIONS)
lda = LDA(n_components=DIMENSIONS)
print(data.columns)

y = data["variety"]
# DAY_THRESHOLD = 30
# x_data_slice = x_data.loc[data["day"] > DAY_THRESHOLD]
# data = data.loc[data["day"] > DAY_THRESHOLD]
# varieties: 'กข43' 'กข79' 'กข85'  'ปทุมธานี 1'ปทุมธษนี 1
print(data["type exp"].unique())
# data = data.loc[data["variety"] == 'กข43']
print(data["type exp"].unique())
x_data = data[x_columns]

x_pca = pca.fit(x_data).transform(x_data)
x_lda = lda.fit(x_data, data[LDA_COLUMN]).transform(x_data)
if ADD_DEAD_LEAVES:
    dead_x_pca = pca.transform(dead_leaf_data[x_columns])
colors = ["navy", "turquoise", "darkorange", "magenta"]
# color_map = {'กข43':"navy", 'กข79': "turquoise",
#              'กข85': "darkorange", 'ปทุมธานี 1': "magenta"}
# data["color"] = data["variety"].map(color_map)
print(names)
lw = 2

plt.figure()

for color, name in zip(colors, names):
    for exp in type_exps:
        marker = marker_map[exp]
        ser1 = data["variety"] == name
        ser2 = data["type exp"] == exp

        mask = ser1 & ser2
        print(name, exp)
        print(x_pca[mask, 0].shape)
        print(data['variety'].unique())
        if DIMENSIONS == 2:
            plt.scatter(x_pca[mask, 0], x_pca[mask, 1], color=color,
                        alpha=0.5, label=f"{name} {exp}", marker=marker)
        elif DIMENSIONS == 1:
            plt.hist(x_pca[mask, 0], bins=30,
                     alpha=0.5, label=f"{name} {exp}",
                     color=color, cumulative=CUMULATIVE,
                     histtype='step')
if ADD_DEAD_LEAVES:
    if DIMENSIONS == 2:
        plt.scatter(dead_x_pca[:, 0], dead_x_pca[:, 1], color="black",
                    alpha=0.8, label=f"dead leaves", marker='X',
                    s=20)
    elif DIMENSIONS == 1:
        plt.hist(dead_x_pca[:, 0], bins=30,
                 alpha=0.8, label=f"{name} {exp}",
                 color="black", cumulative=CUMULATIVE)
# plt.xlim([-1000, 100])
plt.legend(prop={'size': 12})
plt.title("PCA of rice control rice leaves")

plt.figure()
for color, name in zip(colors, names):
    for exp in type_exps:
        marker = marker_map[exp]
        ser1 = data["variety"] == name
        ser2 = data["type exp"] == exp
        mask = ser1 & ser2
        if DIMENSIONS == 2:
            plt.scatter(x_lda[mask, 0], x_lda[mask, 1], color=color,
                        alpha=0.5, label=f"{name} {exp}", marker=marker)
        elif DIMENSIONS == 1:
            plt.hist(x_lda[mask, 0], bins=30,
                     alpha=0.8, label=f"{name} {exp}",
                     color=color, cumulative=CUMULATIVE,
                     histtype='step', ls=LINESTYLES[exp])
if ADD_DEAD_LEAVES:
    if DIMENSIONS == 2:
        plt.scatter(dead_x_pca[:, 0], dead_x_pca[:, 1], color="black",
                    alpha=0.8, label=f"dead leaves", marker='X',
                    s=20)
    elif DIMENSIONS == 1:
        plt.hist(dead_x_pca[:, 0], bins=30,
                 alpha=0.8, label=f"dead leaves",
                 color="black", cumulative=CUMULATIVE)
plt.legend(prop={'size': 12})
plt.title("LDA of rice control rice leaves")
plt.show()
