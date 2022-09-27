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
# local files
import get_data

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0

pd.set_option('display.max_columns', None)
# data = pd.read_excel("2022-08-06_AS7265x_UT_rice.xlsx", keep_default_na=False)
# data = pd.read_excel("2022-08-10_AS7265x_UT_rice.xlsx")
# data, wavelenghts = get_data.get_data("08-14")
data = pd.read_excel("w chloro raw first set.xlsx")
# data = pd.read_excel("first_4_days.xlsx")
data = data.dropna(axis='index', how='all')
data = data.fillna(method="ffill")

x_columns = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
# data = data[data["type exp"] !='control']
x_data = data[x_columns]
marker_map = {'X': "control", 'o': "งดน้ำ"}
marker_map = {"control": 'x', "งดน้ำ": 'o'}
# data['marker'] = data['type exp'].map(marker_map)
type_exps = data["type exp"].unique()[1:]  # skip the nan from reference
type_exps = ['control', 'งดน้ำ']
print('======')
print(type_exps)
# print(data["หมายเลขกระถาง"].unique())
# print(data["การทดลอง"].unique())
# print(data["พันธุ์ข้าว"].unique())

names = data["variety"].unique()
if len(names) >= 5:
    names = names[1:]
# data = data[data["variety"] == names[3]]

pca = PCA(n_components=2)
lda = LDA(n_components=2)
print(data.columns)

y = data["variety"]
DAY_THRESHOLD = -2
x_data_slice = x_data.loc[data["day"] > DAY_THRESHOLD]
data = data.loc[data["day"] > DAY_THRESHOLD]

y_slice = data["variety"]
# x_pca = pca.fit(x_data).transform(x_data)
# x_lda = lda.fit(x_data, y).transform(x_data)
print(x_data_slice)
print(data["day"])

x_pca = pca.fit(x_data).transform(x_data_slice)
x_lda = lda.fit(x_data, y).transform(x_data_slice)
y = y_slice
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
        print(f"marker: {marker}")
        print(data["type exp"] == exp)
        ser1 = y_slice == name
        ser2 = data["type exp"] == exp
        print(f"aa: {type(ser1)}, {type(ser2)}")
        mask = ser1 & ser2
        print(mask)
        print(data["type exp"] == exp)
        print(x_pca[mask, 0].shape)
        plt.scatter(x_pca[mask, 0], x_pca[mask, 1], color=color,
                    alpha=0.8, label=f"{name} {exp}", marker=marker)
        # plt.scatter(x_pca[mask, 0], x_pca[mask, 1], color=color,
        #             alpha=0.8, label=name, marker=marker)
# plt.xlim([-1000, 100])
plt.legend(prop={'size': 20})
plt.title("PCA of rice control rice leaves")

plt.figure()
for color, name in zip(colors, names):
    plt.scatter(x_lda[y == name, 0], x_lda[y == name, 1],
                color=color, alpha=0.8, label=name)
# plt.xlim([-20, 0])
plt.legend(prop={'size': 20})
plt.title("LDA of rice control rice leaves")
plt.show()
