# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
# local files
import get_data
import processing

# font_manager = fm.
# font_files = fm.findSystemFonts(fontpaths=["/resources/fonts"])
# font_list = fm.createFontList(['THSarabunNew.ttf'])
# ff = fm.fontManager.findfont('THSarabunNew.ttf')
# print(ff)
fm.fontManager.addfont('THSarabunNew.ttf')
# fm.fontManager.ttflist.extend(font_list)
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
ALPHA = 0.9
# "การทดลอง"- type of experiment , "พันธุ์ข้าว" - rice variety, "หมายเหตุ" - notes
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                         constrained_layout=True)
data, wavelengths = get_data.get_data("09-07")
# print(data)
leaf_nums = data['Leaf number'].unique()
print(leaf_nums)
leaf_exp_map = {}
leaf_var_map = {}
for leaf_num in leaf_nums:
    # print(data[data['Leaf number'] == leaf_num])
    df_slice = data[data['Leaf number'] == leaf_num]
    leaf_exp_map[leaf_num] = df_slice["type exp"].unique().tolist()[0]
    leaf_var_map[leaf_num] = df_slice["variety"].unique().tolist()[0]
print(leaf_exp_map)
print(leaf_var_map)

print('+=====+=+')
print(data)
print(data.columns)
print('----')
data = data.groupby('Leaf number', as_index=False).mean()
data["type exp"] = data["Leaf number"].map(leaf_exp_map)
data["variety"] = data["Leaf number"].map(leaf_var_map)
print(data)
print(data.columns)
names = data["variety"].unique()[1:]
colors = ["navy", "turquoise", "darkorange", "magenta"]
# names = data["พันธุ์ข้าว"].unique()

x_columns = []
data = data[data['variety'] != "กระดาษขาว"]
# data = data.groupby('variety', as_index=False).mean()
data = data.groupby(["variety", "type exp"], as_index=False).mean()
y = data["variety"]
# ['กระดาษขาว' 'กข43' 'กข79' 'กข85' 'ปทุมธานี 1']
# data = data[data['variety'] == 'กข85']
print(data)

print('[[[[')
print(data['variety'].unique())
print(data)
print(names)
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = data[x_columns]

# axes[0].plot(wavelengths, x_data.T)
axes.set_title("Raw Data")
axes.set_ylabel("Reflectance")
for color, name in zip(colors, names):
    print(name)
    ser1 = y == name
    for ls, exp_type in zip(['solid', 'dashed'], ['control', 'งดน้ำ']):
        ser2 = data["type exp"] == exp_type
        mask = ser1 & ser2
        # print(wavelengths)
        # print(x_data[mask].T)
        _y = x_data[mask].values.tolist()[0]
        _y = x_data.loc[mask].T.values[:, :]
        print(_y)
        print(len(wavelengths), len(_y))
        print(type(wavelengths), type(_y))
        # axes.plot(wavelengths, _y)
        axes.plot(wavelengths, x_data.loc[mask].T.values[:,:], ls=ls,
                  color=color, label=f"{name}, {exp_type}",
                  alpha=ALPHA)

# handles, labels = axes[0].get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes[0].legend(by_label.values(),
#                by_label.keys(),
#                prop={'size': 20})
axes.legend(prop={'size': 20})
# SNV Data
# snv_data = processing.snv(x_data)
# axes[1].set_title("SNV Data")
# axes[1].set_ylabel("Reflectance")
# for color, name in zip(colors, names):
#     axes[1].plot(wavelengths, snv_data[y == name].T,
#                  color=color, label=name, alpha=alpha)
#
# msc_data, _ = processing.msc(x_data)
# axes[2].set_title("MSC Data")
# axes[2].set_ylabel("Reflectance")
# for color, name in zip(colors, names):
#     axes[2].plot(wavelengths, msc_data[y == name].T,
#                  color=color, label=name, alpha=alpha)

plt.show()
