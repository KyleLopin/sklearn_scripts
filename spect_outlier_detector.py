# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.svm import OneClassSVM
# local files
import processing

as7262_wavelengths = ["450 nm", "500 nm", "550 nm",
                      "570 nm", "600 nm", "650 nm", ]


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def MahalanobisDist(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            vars_mean = []
            for i in range(data.shape[0]):
                vars_mean.append(list(data.mean(axis=0)))
            diff = data - vars_mean
            md = []
            for i in range(len(diff)):
                md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

            if verbose:
                print("Covariance Matrix:\n {}\n".format(covariance_matrix))
                print("Inverse of Covariance Matrix:\n {}\n".format(inv_covariance_matrix))
                print("Variables Mean Vector:\n {}\n".format(vars_mean))
                print("Variables - Variables Mean Vector:\n {}\n".format(diff))
                print("Mahalanobis Distance:\n {}\n".format(md))
            return md
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")


def zscore(array):
    zscore = (array - np.median(array)) / array.std()
    print('====')
    print(np.where(np.abs(zscore) > 5))


def as7262_outliers(data, scatter_correction=None):
    data_columns = data[as7262_wavelengths]
    print(data_columns)
    # data_columns.T.plot()
    # plt.plot(data_columns.T)
    plt.show()
    if scatter_correction == "SNV":
        data_columns = processing.snv(data_columns)
    elif scatter_correction == "MSC":
        data_columns, _ = processing.msc(data_columns)

    # svm = OneClassSVM().fit_predict(snv_data)
    # print(svm)
    robust_cov = MinCovDet().fit(data_columns)
    mahal_dist = robust_cov.mahalanobis(data_columns)
    # mahal_dist = MahalanobisDist(np.array(data_columns), verbose=True)
    print(mahal_dist)


    zscore(data_columns)
    print('+++++')
    mean = np.mean(mahal_dist)
    std = 3*np.std(mahal_dist)
    print(mean, std)
    print(mean - std, mean + std)
    zscore_mahal = (mahal_dist - mean) / np.std(mahal_dist)
    # print(zscore_mahal)
    # print(zscore_mahal.max(), zscore_mahal.argmax(), data_columns.loc[zscore_mahal.argmax()])
    print('pppp')
    print(data_columns)
    print(zscore_mahal.argmax())
    outliers = data_columns.loc[zscore_mahal > 3].index
    outliers = data_columns.iloc[zscore_mahal.argmax()].name
    # print(data_columns.loc[zscore_mahal > 3].index)
    rows = data_columns.loc[outliers]
    # print(data_columns.loc[zscore_mahal.argmax()].name)
    print(data_columns.shape)
    print(rows)

    # print((mahal_dist-mahal_dist.mean()).std())
    # print(mahal_dist.std())
    # print(mahal_dist.mean() + 3*mahal_dist.std())
    # mahal_dist2 = MahalanobisDist(np.array(data_columns), verbose=True)
    n, bins, _ = plt.hist(zscore_mahal, bins=40)
    plt.show()

    # x_hist = np.linspace(min(mahal_dist), max(mahal_dist), 100)
    #
    # popt, pcov = curve_fit(gauss_function, bins[:len(n)], n, maxfev=100000, p0=[300, 0, 20])
    # new_fit = gauss_function(x_hist, *popt)
    # plt.plot(x_hist, new_fit, 'r--')
    # color = data_columns.shape[0] * ["#000000"]
    # color[data_columns.loc[zscore_mahal.argmax()].name] = "#FF0000"
    plt.plot(data_columns.T, c="black")
    plt.plot(rows.T, c="red")
    plt.plot(data_columns.mean(), c="blue", lw=4)
    # snv_data.T.plot(color=color)
    plt.show()


if __name__ == "__main__":
    data_ = pd.read_csv("Jasmine AS7262.csv")
    print(data_.shape)
    # data_ = data_.loc[data_["LED current"] == "25 mA"]

    print(data_.shape)
    data_ = data_.drop([2040])

    as7262_outliers(data_, scatter_correction="MSC")
