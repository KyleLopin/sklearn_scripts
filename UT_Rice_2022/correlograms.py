# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautus Lopin"

# installed libraries
import os

# installed libraries
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns

print(plt.rcParams.keys())
fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 16.0
plt.rcParams['ytick.labelsize'] = 16.0
plt.rcParams['axes.labelsize'] = 20.0
plt.rcParams['figure.titlesize'] = 20.0
plt.rcParams['legend.fontsize'] = 16.0
plt.rcParams['legend.title_fontsize'] = 16.0

SET = "first"
SENSOR = "AS7265x"
TYPE = "reflectance"
N_COMPS = 2
KERNEL = 'linear'
Y_NAME = "type exp"

# dead_leaf_data = pd.read_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
# dead_leaf_data = dead_leaf_data.loc[dead_leaf_data["health"] == 0]
dead_leaf_data = pd.read_excel(f"dead_leaves_{SENSOR}_{TYPE}.xlsx")
all_data = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")

x_columns = []
for column in dead_leaf_data.columns:
    if 'nm' in column:
        x_columns.append(column)

full_data = pd.concat([all_data, dead_leaf_data])
full_data = full_data.loc[full_data['type exp'] != 'reference']
full_data = full_data.loc[full_data['variety'] != 'กระดาษขาว']
print(full_data.columns)
print(full_data['type exp'].unique())
x_data = full_data[x_columns]
y = full_data[Y_NAME]
print(full_data['variety'].unique())

pca = KernelPCA(n_components=N_COMPS, kernel=KERNEL)
# pca = LDA(n_components=2)
columns_ = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
Xpca = pca.fit_transform(x_data, y)
df = pd.DataFrame(Xpca, columns=columns_[:N_COMPS])
print(df)
# df["TOC"] = (N_COMPS*y.values).T.astype("int")
df[Y_NAME] = y.values.T

pg = sns.pairplot(df, hue=Y_NAME, palette="OrRd")  #
pg = pg.map_diag(sns.kdeplot)
pg.map_lower(sns.kdeplot, levels=2, color=".2")
name = f"{SENSOR} {KERNEL} PCA by {Y_NAME} for {SET} set"
plt.suptitle(name)
plt.gcf().subplots_adjust(top=0.92, bottom=0.1, left=0.12)
plt.savefig(os.path.join("correlograms", name+".tiff"))
plt.show()
