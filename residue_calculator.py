# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

chloro_columns = ['Avg Total Chlorophyll (µg/cm2)', 'Avg Chlorophyll a (µg/cm2)',
                  'Avg Chlorophyll b (µg/cm2)']
chloro_columns = ['Avg Total Chlorophyll (µg/cm2)']


# drop_indexes = [212, 197, 248, 224, 238,  # rice
#                 218, 153, 245, 105, 113,
#                 293, 297, 152]
# drop_indexes = [49, 72, 82, 108, 223]  # sugarcane
# drop_indexes = [48, 76, 250, 60, 24,  # banana
#                 43, 238, 155, 240, 29,
#                 134, 295, 46, 70, 150,
#                 172, 100, 120, 15, 257,
#                 65, 9, 174, 249, 96,
#                 213]
drop_indexes = [238, 8, 11, 217, 136,  # mango
                37, 133, 220, 59]
# drop_indexes = [42, 292, 154, 119, 78]  # jasmine
# drop_indexes = [238]
drop_leafs = [66]
drop_leafs = []

an_pos = (3, 50)


plt.style.use('bmh')


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def fix_chloro_file(filename, new_filename):
    data = pd.read_csv(filename)
    print(data)
    filled_data = data.fillna(method='ffill')
    print(filled_data)
    filled_data.to_csv(new_filename)


def calc_residues(data):
    print(data)
    print(data.columns)
    for c_col in chloro_columns:
        true_value = data[c_col]
        print(true_value)
        print(c_col)


def plot_data_pts(data):
    print(data)
    print(data.columns)
    plt.scatter(data['Avg Total Chlorophyll (µg/cm2)'], data['Total Chlorophyll (µg/cm2)'])

    y = data['Avg Total Chlorophyll (µg/cm2)']
    x = data['Total Chlorophyll (µg/cm2)']

    # cov = EllipticEnvelope().fit(y.values.reshape(-1, 1))
    cov = IsolationForest().fit(y.values.reshape(-1, 1))
    print(cov.predict(y.values.reshape(-1, 1)))
    data['predict'] = cov.predict(y.values.reshape(-1, 1))
    print(data)
    print(data.loc[data['predict'] == -1]['Leaf No.'])
    outliers_x = data.loc[data['predict'] == -1]['Avg Total Chlorophyll (µg/cm2)']
    outliers_y = data.loc[data['predict'] == -1]['Total Chlorophyll (µg/cm2)']
    plt.scatter(outliers_x, outliers_y, marker='x', s=50)

    print('mea: ', mean_absolute_error(x, y))
    print('r2: ', r2_score(x, y))

    plt.show()


def outlier_finder(data):
    print(data)
    print(data.columns)
    detector = LocalOutlierFactor(n_neighbors=2)
    # print(data['Total Chlorophyll (µg/cm2)'])
    # print(data['Leaf No.'].unique())
    for leaf in data['Leaf No.'].unique():
        print(leaf)
        # print(data.loc[data['Leaf No.'] == leaf])
        # print(data.loc[data['Leaf No.'] == leaf]['Total Chlorophyll (µg/cm2)'])
        x = data.loc[data['Leaf No.'] == leaf]['Total Chlorophyll (µg/cm2)']

        x_r = detector.fit_predict(x.values.reshape(-1, 1))
        print(x)
        print(x_r)
        if 0 in x_r:
            print('========')


def plot_res_hist(data):
    new_df = pd.DataFrame()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(8, 4))

    x = data['Avg Total Chlorophyll (µg/cm2)']
    y = data['Total Chlorophyll (µg/cm2)']
    ax1.scatter(x, y, marker='x', c='k', s=20, alpha=0.6,
                lw=1, label="Full data set")

    ax1.set_xlabel(u"Average Measured Chlorophyll (µg/cm\u00B2)")
    ax1.set_ylabel(u"Individual Measured Chlorophyll (µg/cm\u00B2)")

    data = data.drop(drop_indexes)
    mea_orig = mean_absolute_error(x, y)
    r2_orig = r2_score(x, y)
    print('mea: ', mea_orig)
    print('r2: ', r2_orig)


    # print(data)
    # print('------')
    # recalculate Avg total chlorophyll
    avg_total_area = data.groupby('Leaf No.', as_index=True).mean()['Total Chlorophyll (µg/cm2)']
    avg_a_area = data.groupby('Leaf No.', as_index=True).mean()['Chlorophyll a (µg/cm2)']
    avg_b_area = data.groupby('Leaf No.', as_index=True).mean()['Chlorophyll b (µg/cm2)']

    avg_total_wt = data.groupby('Leaf No.', as_index=True).mean()['Total Chlorophyll (µg/mg)']
    avg_a_wt = data.groupby('Leaf No.', as_index=True).mean()['Chlorophyll a (µg/mg)']
    avg_b_wt = data.groupby('Leaf No.', as_index=True).mean()['Chlorophyll b (µg/mg)']

    new_df['Avg Total Chlorophyll (µg/cm2)'] = avg_total_area
    new_df['Avg Chlorophyll a (µg/cm2)'] = avg_a_area
    new_df['Avg Chlorophyll b (µg/cm2)'] = avg_b_area

    new_df['Avg Total Chlorophyll (µg/mg)'] = avg_total_wt
    new_df['Avg Chlorophyll a (µg/mg)'] = avg_a_wt
    new_df['Avg Chlorophyll b (µg/mg)'] = avg_b_wt

    # print(type(avg_total_area))
    # print(avg_total_area)
    # data.loc[data.index, 'Avg Total Chlorophyll (µg/cm2)'] = avg

    for index, row in data.iterrows():
        # print('++++++++')
        # print(row)
        # print(type(row))
        leaf_num = row['Leaf No.']
        # print(row['Leaf No.'])
        # print("New avg: ", avg.loc[leaf_num])
        # print('AVG old: ', row['Avg Total Chlorophyll (µg/cm2)'])
        # row['Avg Total Chlorophyll (µg/cm2)'] = avg.loc[leaf_num]
        data.loc[index, 'Avg Total Chlorophyll (µg/cm2)'] = avg_total_area.loc[leaf_num]
        data.loc[index, 'Avg Chlorophyll a (µg/cm2)'] = avg_a_area.loc[leaf_num]
        data.loc[index, 'Avg Chlorophyll b (µg/cm2)'] = avg_b_area.loc[leaf_num]

        data.loc[index, 'Avg Total Chlorophyll (µg/mg)'] = avg_total_wt.loc[leaf_num]
        data.loc[index, 'Avg Chlorophyll a (µg/mg)'] = avg_a_wt.loc[leaf_num]
        data.loc[index, 'Avg Chlorophyll b (µg/mg)'] = avg_b_wt.loc[leaf_num]

        # print(row)
        # print(data.index[data['Leaf No.'] == row['Leaf No.']], )
        # leaf_no_indices = data.index[data['Leaf No.'] == row['Leaf No.']]
        # data.loc[leaf_no_indices, 'Avg Total Chlorophyll (µg/cm2)'] = avg

    # print(data)
    # data.to_csv("foobar.csv")
    new_df.to_csv("foobar.csv")
    # ham
    y = data['Avg Total Chlorophyll (µg/cm2)']
    x = data['Total Chlorophyll (µg/cm2)']
    residues = y-x
    abs_res = np.abs(residues)
    data['res'] = abs_res
    data['zscores'] = stats.zscore(residues)
    zscores = stats.zscore(residues)
    print('Max: ', data.loc[data['res'].idxmax()])

    # print(zscores.argmax())
    print(abs_res.argmax())
    # print(data.iloc[abs_res.argmax()])
    # print('max: ', abs_res.max(), abs_res.argmax(), zscores[abs_res.argmax()])
    mean = np.mean(residues)
    sigma = np.sqrt(np.var(residues))
    x_hist = np.linspace(min(residues), max(residues), 100)
    print('hist')

    print(np.histogram(residues))
    n, bins, _ = ax2.hist(residues, bins=50)
    print('n:', n)
    print('bins: ', bins)
    print(len(n), len(bins))

    popt, pcov = curve_fit(gauss_function, bins[:len(n)], n)
                           # p0=[20, mean, sigma])
    print('fit data: ', popt)
    new_fit = gauss_function(x_hist, *popt)
    # print(new_fit)
    ax2.plot(x_hist, new_fit, 'r--')
    mea_final = mean_absolute_error(x, y)
    r2_final = r2_score(x, y)
    print('mea: ', mea_final)
    print('r2: ', r2_final)

    ax2.axvline(3 * sigma, ls='--', c='k')
    ax2.axvline(-3 * sigma, ls='--', c='k')
    print(3*sigma)

    ax1.scatter(data['Avg Total Chlorophyll (µg/cm2)'],
                data['Total Chlorophyll (µg/cm2)'],
                marker='o', c='b', s=20,
                alpha=0.2, label="Outliers Removed")
    str_ = f"MEA original = {mea_orig:.3f}\n" \
           f"r\u00B2 original = {r2_orig:.3f}\n" \
           f"MEA final = {mea_final:.3f}\n" \
           f"r\u00B2 final = {r2_final:.3f}"
    ax1.annotate(str_, an_pos)
    ax2.set_xlabel(u"Measured Chlorophyll Residues (µg/cm\u00B2)")
    ax2.set_title("Histogram of residues of measured\n"
                  "chlorophyll data after removing outliers", fontsize=11)
    fig.suptitle("Removing mango chlorophyll outliers", fontsize=14)

    ax1.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # data = pd.read_csv("new_chloro_mango.csv")
    # data = pd.read_excel("Sugarcane Chlorophyll content.xlsx")
    # data = pd.read_excel("new_chloro_rice.xlsx")
    # data = pd.read_csv("new_chloro_rice.csv")
    # print(data)
    # print(data.columns)
    # fix_chloro_file("Mango Chlorophyll content.csv", "new_chloro_mango.csv")
    filename = "new_chloro_mango.csv"
    data = pd.read_csv(filename)
    # plot_data_pts(data)
    # outlier_finder(data)
    print(data.index)
    print(data.shape)
    # for leaf_no in drop_leafs:
    #     data = data[data["Leaf No."] != leaf_no]
    # data = data.drop(drop_indexes)
    print(data.shape)
    plot_res_hist(data)
