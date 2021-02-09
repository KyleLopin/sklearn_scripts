# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed files
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
# local files
import get_data
import processing


BACKGROUND = [687.65, 9453.7, 23218.35, 9845.05, 15496.7,
              18118.55, 7023.8, 7834.1, 28505.9, 4040.9,
              5182.3, 1282.55, 2098.85, 1176.1, 994.45,
              496.45, 377.55, 389.75]
x, y = get_data.get_data("mango", "as7265x", int_time=150,
                             position=2, led="b'White'",
                             led_current="25 mA")
y = y['Avg Total Chlorophyll (µg/cm2)']
x_reflect = x / BACKGROUND
x_snv = processing.snv(x_reflect)
x_msc, _ = processing.msc(x_reflect)
x_robust = RobustScaler().fit_transform(x_msc)
plt.style.use('dark_background')

pls = PLS(n_components=6)
pls.fit(x_msc, y)

