# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Helper functions to make analyze and plot chlorophyll spectrum data
"""

# standard libraries
import os
# installed libraries
import numpy as np
import pandas as pd

__author__ = "Kyle Vitatus Lopin"

# as7262_background = [250271, 176275, 216334, 219763, 230788, 129603]
# column_names_data = ['450.1', '500.1', '550.1', '570.1', '600.1', '650.1']


def make_file_list():
    """ Make a small GUI for user to select all the data files to be analyzed and return the list
    of filenames """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    filenames = filedialog.askopenfilename(initialdir="/", title="Select files to analze",
                                           filetypes=(("xcel files", "*.xlsx"), ("all files", "*.*")))
    print(filenames)



def get_all_files_with_stub(stub, folder):
    """ Find all files in current directory that have the stub in their name """
    named_files = []
    for root, directory, files in os.walk(os.getcwd()):
        print('root', root, folder, folder not in root, directory, files)
        if folder not in root:
            print('PASS')
            print(folder, root, files)
            continue
        print('folder', root, directory)
        for filename in files:
            print('===>', filename)
            if stub in filename and ('.csv' in filename or '.xlsx' in filename):
                print(root, filename)
                print('==================')
                named_files.append(root+'/'+filename)
    return named_files


def check_saturated_data(_dataframe):
    print(_dataframe.index)
    for i, _index in enumerate(_dataframe):
        print(_index)
        print('l=====', _dataframe[_index].tolist())
        if 'Raw data' in _dataframe[_index].tolist():
            print("GOT RAW DATA, ", _index)
            start_raw_index = i
        elif 'Calibrated data' in _dataframe[_index].tolist():

            end_raw_index = i
            print("end raw data", start_raw_index, end_raw_index)


def make_as7265x_data():
    """ Put all data for the as7265x into a single data file """
    as7265x_files = get_all_files_with_stub('as7265x')
    print(as7265x_files)
    starting = True
    for file in as7265x_files:
        print(file)
        excel_file = pd.ExcelFile(file)
        print(excel_file.sheet_names)
        sheet_name = None
        for _sheet_name in excel_file.sheet_names:
            if 'as7265x' in _sheet_name:
                sheet_name = _sheet_name
        if sheet_name:
            file_data = pd.read_excel(file, sheet_name=sheet_name)
        else:  # leaf 1-6 file has error, this fixes it
            file_data = pd.read_excel(file, sheet_name='Sheet1')
        print('================', type(file_data))
        print(file_data)
        if starting:
            all_data = file_data
            starting = False
        else:
            all_data = all_data.append(file_data)
        print(all_data)
    print(all_data)
    all_data.to_csv('as7265x_mango_leaves.csv')


def make_as7262_data(normalized=False, keep_raw=False):
    """ Put all data for the as7262 into a single data file

     :params: normalized: boolean - if true, divide leaf reflectance by
     the background (white styrofoam reflectance) reflectance
     :params: keep_raw - if False remove the raw count reflection numbers
     and leave just the calibrated numbers
     """
    starting = True
    as7262_files = get_all_files_with_stub('as7262')
    for file in as7262_files:
        # print(file)
        try:
            file_data = pd.read_excel(file, sheet_name="Sheet1")
        except:  # sheet name is either Sheet1 or Sheet2
            file_data = pd.read_excel(file, sheet_name="Sheet2")
        if starting:
            all_data = file_data
            starting = False
        else:
            all_data = all_data.append(file_data)
            # print(all_data)

    print('=========')
    all_data.set_index('Leaf number', inplace=True)
    print(all_data)
    chlorophyll_data_filename = get_all_files_with_stub('absorbance')[0]
    chlorophyll_data = pd.read_excel(chlorophyll_data_filename, sheet_name='Summary')
    chlorophyll_data.set_index('Leaf number', inplace=True)
    # print(chlorophyll_data)
    for column_name in chlorophyll_data.columns:
        all_data[column_name] = chlorophyll_data[column_name]

    # all_data['Total Chlorophyll (ug/ml)'] = chlorophyll_data['Total Chlorophyll (ug/ml)']
    # all_data.sort_values(by='Leaf number', inplace=True)
    print(all_data)
    all_data.to_csv("mango_chloro.csv")


def chloro_data_analysis():
    chloro_data_files = get_all_files_with_stub('absorbance')
    print(chloro_data_files)
    for file in chloro_data_files:
        chloro_data = pd.read_excel(file, sheet_name='Sheet1')
    # print(chloro_data)
    last_leaf_number = None
    for index, row in chloro_data.iterrows():
        # print(type(row), row)
        # print(type(row[1]['Leaf number']), row[1]['Leaf number'])
        leaf_number = row['Leaf number']
        print('leaf number:', leaf_number)
        if not np.isnan(leaf_number):
            # last_leaf_number = int(leaf_number)
            last_leaf_number = int(leaf_number)
            print('leaf number:', last_leaf_number)
            # row[1]['Leaf number'] = last_leaf_number

        # chloro_data.loc[index, 'Leaf number'] = last_leaf_number
        chloro_data.loc[index, 'Leaf number'] = 'Leaf: {0}'.format(last_leaf_number)
    # chloro_data['Leaf number'] = chloro_data['Leaf number'].astype(int)
    print(chloro_data)
    summary_data = chloro_data.groupby('Leaf number', as_index=True).mean()
    print(summary_data)
    stdev_data = chloro_data.groupby('Leaf number', as_index=True).std()
    print("SSSSSSSSSSS")
    print(stdev_data)
    return summary_data, stdev_data


def make_mango_summary():
    file_data = pd.read_csv('as7265x_mango_leaves.csv')
    # print(file_data)
    # file_data.set_index('Leaf number', inplace=True)
    # print(file_data.index)
    # led_dataframe = int_time_data_frame.loc[int_time_data_frame['LED'] == led]

    print(file_data['position'].unique())
    # file_data = file_data.loc[file_data['position'] == 'pos 2']
    file_data.reset_index(inplace=True)
    print(file_data)
    for index, row in file_data.iterrows():
        # print(type(row), row)
        # print(type(row[1]['Leaf number']), row[1]['Leaf number'])
        # leaf_number = row['Leaf number'].split(':')[1]
        #
        # file_data.loc[index, 'Leaf number'] = leaf_number

        led_name = row['LED'].lstrip(' ')
        file_data.loc[index, 'LED'] = led_name

    file_data.set_index('Leaf number', inplace=True)
    # file_data.index = file_data.index.astype(int)
    print(file_data)
    sum_data, stdev_data = chloro_data_analysis()
    print('[[[[[[[[[[[[[')
    sum_data.drop('position', axis=1, inplace=True)
    stdev_data.drop('position', axis=1, inplace=True)
    print(sum_data)
    print(']]]]]]]]]')
    print(stdev_data)
    for column_name in sum_data.columns:
        file_data[column_name] = sum_data[column_name]
    for column_name in stdev_data.columns:
        new_column_name = column_name + " stdev"
        file_data[new_column_name] = stdev_data[column_name]
    print(file_data)
    print(sum_data.index)
    print(file_data.index)
    file_data.to_csv('flouro_mango_leaves_full.csv')


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    # print(make_as7262_data())
    # chloro_data_analysis()
    # print(make_as7265x_data())
    make_mango_summary()
    ham
    sum_data, stdev_data = chloro_data_analysis()
    # print(sum_data)
    # print('STD: ')
    # print(stdev_data)
    as7262_data = pd.read_csv('mango_chloro_refl.csv')
    as7262_data.set_index('Leaf number', inplace=True)
    print('=======')
    print(as7262_data)

    print(stdev_data.index)
    # stdev_data.reset_index()
    # for index, row in stdev_data.iterrows():
    #     print(index)
    #     stdev_data.loc[index, 'Leaf number'] = 'Leaf: {0}'.format(index)

    print(stdev_data)

    # stdev_data.set_index('Leaf number')
    print(stdev_data.index)
    # as7262_data.set_index('Leaf number')
    print(as7262_data)
    for column_name in stdev_data.columns:
        print(column_name)
        print(stdev_data[column_name])
        new_column_name = column_name + " stdev"
        as7262_data[new_column_name] = stdev_data[column_name]

        print(as7262_data)
    print(as7262_data.columns)
    print(as7262_data['position'].unique())

    # as7262_data = as7262_data.loc[(as7262_data['position'] == 'pos 1') |
    #                               (as7262_data['position'] == 'pos 2')]
    # print(as7262_data)
    as7262_data.to_csv('mango_chloro_refl3.csv')


def split_train_test_validate(dataframe, validate_samples,
                              test_samples, train_samples=None):
    print(dataframe.shape)
    total_samples_sent = validate_samples + test_samples + train_samples
    if not train_samples:
        train_samples = dataframe.shape[1] - validate_samples - test_samples
    elif dataframe.shape != total_samples_sent:
        raise ValueError("Total samples should equal: {0}, "
                         "you sent {1} total validate, test, "
                         "and training samples".join(dataframe.shape[1],
                                                     total_samples_sent))


