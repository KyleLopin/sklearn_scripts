# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
plt.style.use('seaborn')

area_cols = ['Total Chlorophyll (µg/cm2)', 'Chlorophyll a (µg/cm2)',
             'Chlorophyll b (µg/cm2)']
wt_cols = ['Total Chlorophyll (µg/mg)', 'Chlorophyll a (µg/mg)',
           'Chlorophyll b (µg/mg)']


def chloro_graph_means_and_pts(_data, style='per area'):
    fig, axes, = plt.subplots(2, 2, figsize=(7.5, 9))
    axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    measure_data = _data.drop(238, axis=0)
    print('----')
    print(_data)
    if style == 'per area':
        fig.suptitle("Chlorophyll measurements per area (µg/cm\u00B2)", size=14)
        # mean_area = _data.groupby('Leaf number', as_index=True).mean()
        # print(mean_area)

        for i, col in enumerate(area_cols):
            print('Avg '+col)
            axes[i].scatter(_data['Avg '+col], _data[col])
            # print(_data.index)
            # print(_data.iloc[237])

            line_model = LinearRegression()
            Y = np.array(measure_data[col]).reshape(-1, 1)
            X = np.array(measure_data['Avg '+col]).reshape(-1, 1)
            line_model.fit(Y, X)
            y_predict = line_model.predict(X)
            mean = mean_absolute_error(Y, y_predict)
            print(mean)
            axes[i].set_xlabel("Measured average chlorophyll levels")
            axes[i].set_ylabel("Measured average chlorophyll points")

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("new_chloro_mango.csv")
    print(data)
    print(data.columns)
    print(data.iloc[238])

    chloro_graph_means_and_pts(data)

