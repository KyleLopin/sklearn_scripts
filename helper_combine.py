# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Combine spectrum data sets with different light inputs into a single data set.
i.e.
leaf    450 nm      550 nm      LED             integration time
1       120         110         405 nm LED          150
1       150         20          455 nm LED          200

into;
leaf    (450 nm, 405 nm LED, int: 150), (450 nm, 455 nm LED, int: 200)  (550 nm, 405 nm LED, int: 150), (550 nm, 455 nm LED, int: 150)
1       120                                 150                                 110                          20
"""

__author__ = "Kyle Vitatus Lopin"


import pandas as pd

data = pd.read_csv('flouro_mango_leaves_full.csv')

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)

print(data.columns)
print(data['Leaf number'])
print(data['Leaf number'].unique())
print(data.index)

# for leaf_number in data['Leaf number'].unique():
#     leaf_data = data[data['Leaf number'] == leaf_number]
#     print(leaf_data.shape)
#
#     for data_column in data_columns:
#         print(data_column)
#         new_column_name = "{0}, int: {1}, LED: {2}"

leaf_numbers = data['Leaf number'].unique()
int_times = data['integration time'].unique()
leds = data['LED'].unique()

print(int_times)
print(leds)
print(data_columns)


print('========')
# new_data_frame['410 nm, White LED, int: 150'][55] = 55
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 500)
new_dataframe = pd.DataFrame()
start = True
for led in leds:
    for int_time in int_times:
        sub_df = data[((data['LED'] == led) | (data['LED'] == led)) &
                      (data['integration time'] == int_time)]
        # print(sub_df)
        # print(sub_df['Leaf number'].unique().shape)
        # print('2:', data['LED'].unique(), data['integration time'].unique())
        new_column_names = dict()
        for data_column in sub_df.columns:
            if ' nm' in data_column:
                new_column_names[data_column] = "{0}, {1}, int: {2}".format(data_column,
                                                                            led, int_time)
        # print('=========')
        # print(new_column_names)
        sub_df.rename(columns=new_column_names, inplace=True)
        # print('=========')
        # print(sub_df['Leaf number'] + ' ' + sub_df['position'])
        # sub_df['Measurement'] = 'Leaf:' + str(sub_df['Leaf number']) + ' ' + 'pos' + str(sub_df['position'])
        sub_df['Measurement'] = sub_df['Leaf number'] + ' ' + sub_df['position']

        sub_df.set_index('Measurement', inplace=True)
        # print(sub_df.index.unique())
        # print(new_dataframe.index.unique())
        print('1: ', new_dataframe.shape, sub_df.shape, led, int_time, len(leds), int_time.shape)
        # print(sub_df.columns)
        # print(list(sub_df.index))
        # print(list(new_dataframe.index))
        index_sub = list(sub_df.index)
        index_data = list(new_dataframe.index)
        # for index_name in index_sub:
        #     if index_name not in index_data:
        #         print("Missing index:", index_name)
        sub_df.to_csv("test.csv")

        new_dataframe = pd.concat([new_dataframe, sub_df], axis=0)
        # new_dataframe = new_dataframe.drop_duplicates(keep="first")
        new_dataframe = new_dataframe.loc[:, ~new_dataframe.columns.duplicated()]
        # print('+++++++')
        # print(new_dataframe)


# print(new_dataframe)
print(new_dataframe.columns)
# print(new_dataframe['LED'])
# print(new_dataframe['LED'])
new_dataframe.to_csv("test.csv")

