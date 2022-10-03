# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
import seaborn as sns

SET = "first"
SENSOR = "AS7265x"
TYPE = "reflectance"
N_COMPS = 2
KERNEL = 'linear'
Y_NAME = "type exp"

dead_leaf_data = pd.read_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
dead_leaf_data = dead_leaf_data.loc[dead_leaf_data["health"] == 0]
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
columns_ = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
Xpca = pca.fit_transform(x_data, y)
df = pd.DataFrame(Xpca, columns=columns_[:N_COMPS])
print(df)
# df["TOC"] = (N_COMPS*y.values).T.astype("int")
df[Y_NAME] = y.values.T

pg = sns.pairplot(df, hue=Y_NAME, palette='OrRd')
pg = pg.map_diag(sns.kdeplot)
plt.suptitle(f"{KERNEL} PCA by {Y_NAME}")
plt.gcf().subplots_adjust(top=0.92, bottom=0.1, left=0.08)
plt.show()
