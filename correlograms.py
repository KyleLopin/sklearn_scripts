# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
import seaborn as sns

import get_data


chloro_types = ['Chlorophyll a (µg/mg)', 'Chlorophyll b (µg/mg)',
                'Total Chlorophyll (µg/mg)']
# y_name = chloro_types[2]
y_name = 'Avg Total Chlorophyll (µg/cm2)'
x_data, y, data = get_data.get_data("mango", "as7262", int_time=[150],
                                   position=[1, 2, 3], led_current=["25 mA"], return_type="XYZ")
print(x_data)
print(y)
y = y[y_name]
n_comps = 2
kernel = 'rbf'
pca = PCA(n_components=n_comps)
pca = KernelPCA(n_components=n_comps, kernel=kernel)
Xpca = pca.fit_transform(x_data)

columns_ = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
df = pd.DataFrame(Xpca, columns=columns_[:n_comps])
print(y.shape)
print(y)
print(y.min(), y.max())
bins = np.linspace(y.min(), y.max(), 7)
# print(10*y.values.T.astype("int"))
# print(len(10*y.values.T.astype("int")))
# print(df.shape)
df["TOC"] = (n_comps*y.values).T.astype("int")
print(df)
pg = sns.pairplot(df, hue="TOC", palette='OrRd')
# pg = sns.PairGrid(df, hue="TOC", palette='OrRd')
pg = pg.map_diag(sns.kdeplot)
# pg = pg.map_lower(sns.kdeplot, cmap="seismic", shaded=True)
# pg = pg.map_upper(plt.scatter, hue="TOC", palette='OrRd')

plt.gcf().suptitle("{0} Kernel PCA Pairplots"
                   "".format(kernel), size=24)

plt.gcf().subplots_adjust(top=0.94, bottom=0.07, left=0.05)
plt.show()
