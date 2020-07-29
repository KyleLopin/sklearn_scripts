# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
import seaborn as sns

import data_getter


chloro_types = ['Chlorophyll a (ug/ml)', 'Chlorophyll b (ug/ml)',
                'Total Chlorophyll (ug/ml)']
y_name = chloro_types[2]
x_data, y, data = data_getter.get_data('as7263 betal')
print(x_data)
print(y)
y = y[y_name]
n_comps = 5
kernel = 'rbf'
pca = PCA(n_components=n_comps)
pca = KernelPCA(n_components=n_comps, kernel=kernel)
Xpca = pca.fit_transform(x_data)

columns_ = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
df = pd.DataFrame(Xpca, columns=columns_[:n_comps])
print(y.shape)
print(y)
print(y.min(), y.max())
bins = np.linspace(y.min(), y.max(), 6)
print(10*y.values.T.astype("int"))
print(len(10*y.values.T.astype("int")))
print(df.shape)
df["TOC"] = (5*y.values).T.astype("int")
print(df)
pg = sns.pairplot(df, hue="TOC", palette='OrRd')
# pg = sns.PairGrid(df, hue="TOC", palette='OrRd')
pg = pg.map_diag(sns.kdeplot)
pg = pg.map_lower(sns.kdeplot, cmap="seismic", shaded=True)
# pg = pg.map_upper(plt.scatter, hue="TOC", palette='OrRd')

plt.gcf().suptitle("{0} Kernel PCA Pairplots"
                   "".format(kernel), size=24)

plt.gcf().subplots_adjust(top=0.94, bottom=0.07, left=0.05)
plt.show()
