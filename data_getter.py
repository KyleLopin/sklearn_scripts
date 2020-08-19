# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import pandas as pd


def get_data(leaf_type: str, remove_outlier=False, only_pos2=False):
    chloro_data = None
    AS7263_channel_list = ["610 nm", "680 nm", "730 nm",
                           "760 nm", "810 nm", "860 nm"]
    AS7263_channel_list_2 = ["610 nm,", "680 nm,", "730 nm,",
                             "760 nm,", "810 nm,", "860 nm,"]
    if leaf_type == 'as7262 mango':
        data = pd.read_csv('as7262_mango.csv')
        if remove_outlier:
            print(data.columns)
            # data = data.drop(["Leaf: 49"])
            # print(data[data["Leaf number"] == "Leaf: 49"])
            data = data[data["Leaf number"] != "Leaf: 54"]
            data = data[data["Leaf number"] != "Leaf: 41"]
            # data = data[data["Leaf number"] != "Leaf: 37"]
            # data = data[data["Leaf number"] != "Leaf: 38"]
            # data = data[data["Leaf number"] != "Leaf: 33"]
            # data = data[data["Leaf number"] != "Leaf: 34"]
            # data = data[data["Leaf number"] != "Leaf: 35"]


        data = data.loc[data['integration time'] == 150]
        data = data.loc[data['position'] == 'pos 2']
        data = data.loc[data['Total Chlorophyll (ug/ml)'] < .6]
        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == 'as7263 mango':
        data = pd.read_csv('as7265x_mango_leaves.csv')
        # print(data)
        # print('=======')
        channel_data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED"]

        chloro_columns = []
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        channel_data_columns.extend(chloro_columns)
        data = data[channel_data_columns]
        # data = data.groupby(['Leaf number', 'LED']).mean()
        print(data)
        print('======')

    elif leaf_type == 'as7263 mango verbose':
        data = pd.read_csv('as7265x_mango_rows.csv')
        print(data)
        as7263_channels = ["Leaf number",
                           'Chlorophyll a (ug/ml)',
                           'Chlorophyll b (ug/ml)',
                           'Total Chlorophyll (ug/ml)']
        for column in data.columns:
            print(column)
            for led in AS7263_channel_list_2:
                if led in column:
                    print('=======>', column)
                    as7263_channels.append(column)
        data = data[as7263_channels]
        print(data)
        print(data.columns)

    elif leaf_type == 'as7263 roseapple':
        data = pd.read_csv('as7265x_roseapple_leaves.csv')
        print('=======')
        data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED",
                        'Chlorophyll a (ug/ml)',
                        'Chlorophyll b (ug/ml)',
                        'Total Chlorophyll (ug/ml)',
                        'Chlorophyll a (ug/ml) stdev',
                        'Chlorophyll b (ug/ml) stdev',
                        'Total Chlorophyll (ug/ml) stdev']


        data = data[data_columns]

        channel_columns = []
        chloro_columns = []
        for column in data.columns:
            print(column)
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
            elif 'nm' in column:
                channel_columns.append(column)
        print(chloro_columns)
        chloro_data = data[chloro_columns]
        x_data = data[channel_columns]
        print(data.columns)
        # for index, row in data.iterrows():
        #     leaf = row['Leaf number']
        #     for column in chloro_data.columns:
        #         data.loc[index, column] = chloro_data.loc[leaf, column]
        #     data.append(chloro_data.loc[leaf])
        # data.to_csv("foobar55.csv")

        # data = data.groupby(['Leaf number']).mean()

    elif leaf_type == 'as7263 roseapple verbose':
        data = pd.read_csv('as7265x_roseapple_rows.csv')
        data = data.groupby('Leaf number', as_index=True).mean()

        print(data)
        as7263_channels = ['Chlorophyll a (ug/ml)',
                           'Chlorophyll b (ug/ml)',
                           'Total Chlorophyll (ug/ml)']
        for column in data.columns:
            print(column)
            for led in AS7263_channel_list_2:
                if led in column:
                    print('=======>', column)
                    as7263_channels.append(column)
        data = data[as7263_channels]
        print(data)
        print(data.columns)

    elif leaf_type == 'as7265x mango':
        data = pd.read_csv('as7265x_mango_leaves.csv')

    elif leaf_type == 'as7265x mango verbose':
        data = pd.read_csv('as7265x_mango_rows.csv')
        # data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == 'as7265x roseapple verbose':
        data = pd.read_csv('as7265x_roseapple_rows.csv')
        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == 'as7262 roseapple':
        data = pd.read_csv('as7262_roseapple.csv')
        data = data.loc[(data['current'] == 25)]
        data = data.loc[(data['position'] == 'pos 2')]

        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == "as7265x roseaple":
        pass

    elif leaf_type == 'as7263 roseapple':
        data = pd.read_csv('as7265x_roseapple.csv')
        print(data.columns)
        data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED"]
        chloro_columns = []
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        data_columns.extend(chloro_columns)
        data = data[data_columns]

        chloro_data = pd.read_csv('as7262_roseapple.csv')
        chloro_data = chloro_data.groupby('Leaf number', as_index=True).mean()
        print(chloro_data)
        chloro_columns = []
        print('|||||||||||||||||||||')
        print(chloro_data.columns)
        for column in chloro_data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        print(chloro_columns)
        chloro_data = chloro_data[chloro_columns]
        print(chloro_data)
        print('=====_-------')
        return data, chloro_data

    elif leaf_type == 'as7265x roseapple':
        data = pd.read_csv('as7265x_roseapple.csv')
        print(data.columns)

        chloro_data = pd.read_csv('as7262_roseapple.csv')
        chloro_data = chloro_data.groupby('Leaf number', as_index=True).mean()

        chloro_columns = []
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        chloro_data = chloro_data[chloro_columns]
        return data, chloro_data

    elif leaf_type == 'as7262 betal':
        data = pd.read_csv('as7262_betal.csv')
        data = data.loc[(data['position'] == 'pos 2')]
        data = data.loc[(data['LED current'] == 25)]

        # drops = [81, 86, 249, 254, 297, 302, 321, 326, 393, 398, 609, 614]
        # data = data.drop(index=drops)

        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == 'as7263 betal':
        data = pd.read_csv('as7265x_betal_leaves.csv')
        # print(data.columns)
        print(data.shape)
        print(data['position'].unique())
        data = data.loc[(data['position'] == ' pos 2')]
        print(data.shape)
        data = data.loc[(data['integration time'] == 50)]
        print(data.shape)
        print(data['LED'].unique())
        data = data.loc[(data['LED'] == " White LED")]
        print(data.shape)

        data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED"]
        chloro_columns = []
        x_columns = ["610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm"]
        x_data = data[x_columns]
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        data_columns.extend(chloro_columns)
        data = data[data_columns]

        chloro_data = pd.read_csv('as7262_roseapple.csv')
        chloro_data = chloro_data.groupby('Leaf number', as_index=True).mean()
        # print(chloro_data)
        chloro_columns = []
        # print('|||||||||||||||||||||')
        # print(chloro_data.columns)
        for column in chloro_data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        # print(chloro_columns)
        chloro_data = chloro_data[chloro_columns]
        return x_data, chloro_data, data

    elif leaf_type == "as7262 ylang":
        data = pd.read_csv('as7262_ylang.csv')
        print(data.shape)
        data = data.loc[(data['position'] == 'pos 2')]
        print(data.shape)
        data = data.loc[(data['LED current'] == 0)]
        print(data.shape)
        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == "as7263 ylang":
        data = pd.read_csv('as7265x_ylang_white_led.csv')
        # print(data.columns)
        print(data.shape)
        print(data['position'].unique())
        data = data.loc[(data['position'] == 'pos 2')]
        print(data.shape)
        # data = data.loc[(data['integration time'] == 50)]
        print(data.shape)
        print(data['LED'].unique())
        data = data.loc[(data['LED'] == "White LED")]
        print(data.shape)

        data_columns = ["Leaf number", "integration time",
                        "610 nm", "680 nm", "730 nm",
                        "760 nm", "810 nm", "860 nm",
                        "position", "LED"]
        chloro_columns = []
        x_columns = ["610 nm", "680 nm", "730 nm",
                     "760 nm", "810 nm", "860 nm"]
        x_data = data[x_columns]
        for column in data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        data_columns.extend(chloro_columns)
        data = data[data_columns]

        chloro_data = pd.read_csv('as7262_roseapple.csv')
        chloro_data = chloro_data.groupby('Leaf number', as_index=True).mean()
        # print(chloro_data)
        chloro_columns = []
        # print('|||||||||||||||||||||')
        # print(chloro_data.columns)
        for column in chloro_data.columns:
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
        # print(chloro_columns)
        chloro_data = chloro_data[chloro_columns]
        return x_data, chloro_data, data

        raise NameError("Not finished input")

    elif leaf_type == "as7265x betal":
        data = pd.read_csv('as7265x_betal_leaves.csv')
        print(data.shape)
        print(data['position'].unique())
        data = data.loc[(data['position'] == ' pos 2')]
        print(data.shape)
        data = data.loc[(data['integration time'] == 200)]
        print(data.shape)
        print(data['LED'].unique())
        data = data.loc[(data['LED'] == " White LED")]
        print(data.shape)
        data = data.groupby('Leaf number', as_index=True).mean()

        channel_columns = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm',
                           '560 nm', '585 nm', '610 nm', '645 nm',
                           '680 nm', '705 nm', '730 nm', '760 nm',
                           '810 nm', '860 nm', '900 nm', '940 nm']
        chloro_columns = []
        for column in data.columns:
            print(column)
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
            # elif 'nm' in column:
            #     channel_columns.append(column)
        print(chloro_columns)
        chloro_data = data[chloro_columns]
        x_data = data[channel_columns]

    elif leaf_type == "as7265x ylang":
        data = pd.read_csv('as7265x_ylang_white_led.csv')
        print(data.shape)
        print(data['position'].unique())
        data = data.loc[(data['position'] == 'pos 2')]
        print(data.shape)
        # data = data.loc[(data['integration time'] == 200)]
        # print(data.shape)
        print(data['LED'].unique())
        data = data.loc[(data['LED'] == "White LED")]
        print(data.shape)
        data = data.groupby('Leaf number', as_index=True).mean()

        channel_columns = ['410 nm', '435 nm', '460 nm', '485 nm', '510 nm', '535 nm',
                           '560 nm', '585 nm', '610 nm', '645 nm',
                           '680 nm', '705 nm', '730 nm', '760 nm',
                           '810 nm', '860 nm', '900 nm', '940 nm']
        chloro_columns = []
        for column in data.columns:
            print(column)
            if 'Chlorophyll' in column:
                chloro_columns.append(column)
            # elif 'nm' in column:
            #     channel_columns.append(column)
        print(chloro_columns)
        chloro_data = data[chloro_columns]
        x_data = data[channel_columns]

    elif leaf_type == "new as7262 mango":
        data = pd.read_csv("Mango AS7262_new.csv")
        print(data.shape)
        data = data.loc[(data['LED current'] == "25 mA")]
        print(data.shape)
        data = data.loc[(data['integration time'] == 100)]
        print(data.shape)
        data = data.groupby('Leaf number', as_index=True).mean()

    elif leaf_type == 'as7262 tomato':
        data = pd.read_csv("tomatoes 1-11 AS7262.csv")
        print(data.shape)
        data = data.loc[(data['LED current'] == "25 mA")]
        print(data.shape)
        data = data.loc[(data['integration time'] == 100)]
        print(data.shape)
        data = data.groupby('Fruit number', as_index=True).mean()

    else:
        raise Exception("Not valid input")

    print(leaf_type)
    print(data.columns)
    if only_pos2:
        data = data.loc[(data['position'] == 'pos 2')]

    channel_data_columns = []
    chloro_columns = []
    for column in data.columns:
        if 'nm' in column:
            channel_data_columns.append(column)
        elif 'Chlorophyll' in column:
            chloro_columns.append(column)
    x_data = data[channel_data_columns]
    chloro_data = data[chloro_columns]
    print('---- ', data.columns)
    return x_data, chloro_data, data


if __name__ == '__main__':
    a, b, c = get_data('as7263 roseapple verbose')
    print(a.columns)
    print(b.columns)
    print(c.columns)
