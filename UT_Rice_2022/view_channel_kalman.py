# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Plot each of the channels of the sensor and save the figure into a pdf
"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# local file
import processing

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 15.0
plt.rcParams['ytick.labelsize'] = 15.0
plt.rcParams['axes.labelsize'] = 18.0
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 12

COLORS = ["navy", "turquoise", "darkorange", "magenta"]
ALPHA = 0.8
DATASET = 2
if DATASET == 1:
    SET = "first"
elif DATASET == 2:
    SET = "second"
SENSOR = "AS7265x"
TYPE = "reflectance"
PROCESSING = "565 nm"

