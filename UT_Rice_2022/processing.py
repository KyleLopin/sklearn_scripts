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
        _index = input_data.index
        input_data = input_data.to_numpy()

    for i in range(input_data.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    if _type is pd.DataFrame:
        return pd.DataFrame(data_snv, columns=_columns, index=_index)
    # else return a numpy array
    return data_snv


def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction
    from https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    '''
    # mean centre correction
    _type = type(input_data)

    if _type is pd.DataFrame:
        _columns = input_data.columns
        _index = input_data.index
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
        return pd.DataFrame(data_msc, columns=_columns,
                            index=_index), pd.DataFrame(ref)

    return data_msc, ref


def norm_to_column(input_data, divide_column):
    # print(input_data[divide_column])
    for column in input_data.columns:
        input_data[column] = input_data[column] / input_data[divide_column]
    return input_data.copy()


class KalmanFilter:
    def __init__(self, alpha: float = 0.1, beta: float = 0.2,
                 time_delta: float = 1):
        self.alpha = alpha
        self.beta = beta
        self.time_delta = time_delta

    def estimate_state(self, start_state, _measurement):
        """ After measuring position, estimate current state """
        # print('==', start_state, _measurement)
        est_position = start_state[0] + self.alpha*(_measurement-start_state[0])
        est_velocity = start_state[1] + self.beta*(_measurement-start_state[0])/self.time_delta
        return [est_position, est_velocity]

    def calc_next_state(self, estimated_state):
        """Impliment Kalman filter to estimate new state with a position and 'velocity'"""
        # estimated_state[0] - position; estimated_state[1] - velocity
        new_position = estimated_state[0] + self.time_delta * estimated_state[1]
        new_velocity = estimated_state[1]  # constant velocity
        return [new_position, new_velocity]


def fit_kalman_filter(data: pd.DataFrame, time_column: str,
                      channel: str, alpha: float = 0.6,
                      beta: float = 0.5):
    print(data)
    print(len(data))
    start = data.iloc[0]
    print("start: ", start)
    initial_value = data.iloc[0]
    initial_change = 0
    state = [initial_value, initial_change]
    kalman_filter = KalmanFilter(alpha=alpha, beta=beta)
    states = []
    vel = []
    for time in range(len(data)):
        print('==', state)
        # data_slice = data.loc[data[time_column] == time].mean(numeric_only=True)
        est_state = kalman_filter.estimate_state(state, data.iloc[time])
        state = kalman_filter.calc_next_state(est_state)
        states.append(state[0])
        print(states)
        vel.append(state[1])
    return states, vel
