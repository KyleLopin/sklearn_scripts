# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
# local files
import get_data

plt.style.use('seaborn')

pls = PLSRegression(n_components=6)

x_data, y_data = get_data.get_data("mango", "as7262", int_time=[150],
                                   position=[1, 2, 3], led_current=["25 mA"])
y_data = y_data['Avg Total Chlorophyll (µg/cm2)']
print("===========")
pls.fit(x_data, y_data)
y_predict = pls.predict(x_data)

print(y_predict)
y_predict = pd.DataFrame(y_predict, index=y_data.index)
print(y_predict)

mean_y = y_data.groupby(["Leaf number"]).mean()
mean_y_predict = y_predict.groupby(["Leaf number"]).mean()
print(mean_y_predict)

y_summary = pd.concat([y_predict, mean_y_predict], axis=1)
y_summary.columns = ['predict', 'mean']
y_summary['residue'] = y_summary['predict']-y_summary['mean']
print('ppp')
print(y_summary.to_string())


# plt.scatter(y_summary['predict'], y_summary['mean'])
plt.hist(y_summary['residue'])
print(y_summary['residue'].std())
plt.show()
