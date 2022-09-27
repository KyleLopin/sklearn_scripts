# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

NORMALIZE = True
# ['กระดาษขาว' 'กข43' 'กข79' 'กข85' 'ปทุมธานี 1']
VARIETY = 'กข85'

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0

data = pd.read_excel("w chloro raw set.xlsx")
# data = data.loc[(data["Leaf number"] == 2) |
#                 (data["Leaf number"] == 3) |
#                 (data["Leaf number"] == 4)]
x_columns = []
wavelengths = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = data[x_columns]
type_exps = ['control', 'งดน้ำ']
varieties = data["variety"].unique()
print(varieties)

data = data.loc[data["variety"] == VARIETY]
pots = [1, 2, 3, 4, 5, 6, 7, 8]
colors = ["navy", "turquoise", "darkorange"]
stds = data.groupby(['Leaf number', 'day'], as_index=False).std()
stds = stds[stds["Leaf number"] != 1]
df2 = stds["chloro"].mean()
print(stds)
print(df2)
# print(data)

fig, full_axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 8),
                              constrained_layout=True)
fig.suptitle(f"Vareity: {VARIETY}", fontsize=24)
axes = []
# print(data)
for sub_axes in full_axes:
    axes.extend(sub_axes)
print(axes)
print(data.columns)

for pot, axis in zip(pots, axes):
    sub_data = data.loc[data['pot number'] == pot]
    leaf_nums = sub_data["Leaf number"].unique()
    print(leaf_nums)
    for leaf_num, color in zip(leaf_nums, colors):
        df = sub_data.loc[data["Leaf number"] == leaf_num]
        print(df.shape)
        print(df.columns)
        # print(df)
        if NORMALIZE:
            norm_chloro_df = df[df['day'] == 0]
            print(norm_chloro_df)
            norm_chloro = norm_chloro_df['chloro'].mean()

            print(norm_chloro)
            df.loc['chloro'] = df['chloro']/norm_chloro
        axis.scatter(df['day'], df['chloro'],
                     color=color, label=f"leaf: {leaf_num}")
        axis.set_title(f"Pot number: {pot}")
        # axis.scatter(sub_data.loc[data["Leaf number"] == leaf_num, 'day'],
        #              sub_data.loc[data["Leaf number"] == leaf_num, 'chloro'])
        axis.legend()


plt.show()
