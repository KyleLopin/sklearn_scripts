# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Take in spectrum data with multiple reads and try to identify outliers and remove them.

individual_measurement_outlier - will check if a set of repeated measurements has a standard deviation
amount the measurements that are too high for the set
"""

__author__ = "Kyle Vitautus Lopin"

# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import scale
pd.set_option('display.max_columns', None)

try:  # install thia letters
    fm.fontManager.addfont('THSarabunNew.ttf')
    plt.rcParams['font.family'] = 'TH Sarabun New'
    plt.rcParams['xtick.labelsize'] = 20.0
    plt.rcParams['ytick.labelsize'] = 20.0
except:
    pass  # ttf file into installed

# CONSTANTS
DATASET = "1"
SET = "first"
SENSOR = "AS7265x"
TYPE = "raw"
full_dataset = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
x_columns = []
wavelengths = []
for column in full_dataset.columns:
    if 'nm' in column:
        x_columns.append(column)
        wavelengths.append(column.split()[0])


def check_individual_measurement_outlier_DEPR(df: pd.DataFrame):
    return sum(df[x_columns].std()/df[x_columns].mean())


def calc_zscore(df):
    std = df[x_columns].std(ddof=0)
    mean = df[x_columns].mean()
    z_scores = (df[x_columns] - mean) / std
    if abs(z_scores.max().max()) > 1.7:
        # print(z_scores)
        plt.plot(wavelengths, df[x_columns].T)
        plt.show()


def calculate_measurement_outlier(df):
    for read in df["Read number"]:
        print(read)
        model_df = df.loc[df["Read number"] != read]
        std = model_df[x_columns].std(ddof=0)
        mean = model_df[x_columns].mean()
        plt.hist(std)
        plt.show()
        print(df.loc[df["Read number"] == read, x_columns])
        z_score_col = ((df.loc[df["Read number"] == read, x_columns]
                        - mean) / std)
        print(z_score_col)
        avg_z_score = abs(z_score_col.mean(axis=1).values[0])
        if avg_z_score > 3:
            plt.plot(wavelengths, df[x_columns].T)
            print((df.loc[df["Read number"] == read, x_columns] - mean))
            print(std.T)
            print(df[x_columns])
            plt.show()


def check_sample_measurement_outlier(df: pd.DataFrame):
    robust_cov = MinCovDet().fit(df[x_columns])
    mahal = robust_cov.mahalanobis(df[x_columns])
    plt.hist(mahal)
    plt.show()


if __name__ == "__main__":
    leaves = full_dataset['Leaf number'].unique()
    stds = {}
    for leaf in leaves:
        leaf_df = full_dataset.loc[full_dataset["Leaf number"] == leaf]
        days = leaf_df["day"].unique()
        stds[leaf] = []
        print(f'leaf: {leaf}')
        # check_sample_measurement_outlier(leaf_df)
        for day in days:
            print(f"leave: {leaf}, day: {day}")
            daily_df = leaf_df.loc[leaf_df["day"] == day]
            # stds[leaf].append(check_individual_measurement_outlier(daily_df))
            # calculate_measurement_outlier(daily_df)
            calc_zscore(daily_df)
