# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
# local files
import get_data

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0

alpha = 0.5
pot_map = {1.0: ""}
data, wavelengths = get_data.get_data("08-16")
VARIETIES = ['กระดาษขาว', 'กข43', 'กข79', 'กข85', 'ปทุมธานี 1']
selected_variety = VARIETIES[4]

data = data[data["variety"] == selected_variety]
stored_data = data['type exp'].copy()
data = data.groupby('Leaf number', as_index=False).mean()
x_columns = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = data[x_columns]

print(data)
pot_numbers = data["pot number"].unique()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8),
                         constrained_layout=True)
for pot_num in pot_numbers:
    print(pot_num)
    color = "black"
    label = "dry"
    if pot_num <= 3:
        color = 'aqua'
        label = "control"
    axes[0].plot(wavelengths, x_data[data["pot number"]==pot_num].T,
                 color=color, label=label)
axes[0].set_title(f"{selected_variety} leaves", fontsize=24)
# axes[0].title(f"{selected_variety} leaves", fontdict={'size': 24})
axes[0].set_ylabel("Reflectance")
axes[0].set_xlabel("wavelenght (nm)")
handles, labels = axes[0].get_legend_handles_labels()
print(handles, labels)
by_label = dict(zip(labels, handles))
axes[0].legend(by_label.values(),
               by_label.keys(),
               prop={'size': 18})

# groupby condition
print(data)

data = data.groupby("pot number", as_index=False).mean()
print(data)
x_data = data[x_columns]
for pot_num in pot_numbers:
    color = "black"
    label = "dry"
    if pot_num <= 3:
        color = 'aqua'
        label = "control"
    axes[1].plot(wavelengths, x_data[data["pot number"]==pot_num].T,
                 color=color, label=f"pot number {int(pot_num)}")
axes[1].set_ylabel("Reflectance")
axes[1].set_xlabel("wavelenght (nm)")
axes[1].legend()

print(np.arange(8)//2)
data = data.groupby([0, 0, 0, 1, 1, 1, 1, 1]).mean()
print(data)
x_data = data[x_columns]
for pot_num in pot_numbers:
    color = "black"
    label = "dry"
    if pot_num <= 3:
        color = 'aqua'
        label = "control"
    axes[2].plot(wavelengths, x_data[data["pot number"]==pot_num].T,
                 color=color, label=f"pot number {int(pot_num)}")
axes[2].set_ylabel("Reflectance")
axes[2].set_xlabel("wavelenght (nm)")
axes[2].legend()


plt.show()
