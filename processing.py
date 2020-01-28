# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import numpy as np
import pandas as pd


def snv(input_data):
    """ Standard Normal Variate function to preprocess a pandas dataframe
    core taken from
    https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    :param input_data: pandas Dataframe or numpy array to correct
    :return: SNV corrected data as a pandas Dataframe if a Dataframe was passed in
    or else a numpy array
    """
    _type = type(input_data)
    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    _columns = None
    if type(input_data) is pd.DataFrame:
        _columns = input_data.columns
        input_data = input_data.to_numpy()

    for i in range(input_data.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    if _type is pd.DataFrame:
        return pd.DataFrame(data_snv, columns=_columns)
    # else return a numpy array
    return data_snv


def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction
    from https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    '''
    # mean centre correction
    _type = type(input_data)
    _columns = input_data.columns
    if _type is pd.DataFrame:
        input_data = input_data.to_numpy()
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()
    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)
        # Apply correction
        data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
    if _type is pd.DataFrame:
        return pd.DataFrame(data_msc, columns=_columns), pd.DataFrame(ref)

    return data_msc, ref
