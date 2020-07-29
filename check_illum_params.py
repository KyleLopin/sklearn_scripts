# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import pandas as pd


# filename = "Tomatoes AS7265X.csv"
filename = "A4 paper.csv"
data = pd.read_csv(filename)
print(data)
data_columns = []
for column in data.columns:
    data_columns.append(column)
print(data['led'].unique())
print(data['led current'].unique())

leds = ["b'White'", "b'IR'", "b'UV'",
        "b'White IR'", "b'White UV'",
        "b'UV IR'", "b'White UV IR'"]

currents = ['12.5 mA', '25 mA', '50 mA', '100 mA']
y_name = "Paper number"
int_times = [50, 100, 150]
i = 0

row = data
led1 = "b'White'"
led2 = "b'UV'"
led3 = "b'White UV'"
print(data.columns)
data = data[data[y_name] == "1A"]
print(data)
print('=====')
pd.set_option('display.max_columns', None)
residues = pd.DataFrame()
for current in currents:
    plt.figure(i)
    i += 1
    for int_time in int_times:
        data_1 = data[data["led current"] == current]
        data_1 = data_1[data_1["integration time"] == int_time]
        # data = data[data["led current"] == current]
        print('+++++')

        led1_data = data_1.loc[data_1["led"] == led1]
        led2_data = data_1.loc[data_1["led"] == led2]
        # print(led1_data.columns)
        # print(current, int_time)
        print('led 1', led1)
        print(led1_data)
        print('led 2', led2)
        print(led2_data)
        joint_leds = data_1.loc[data_1["led"] == led3]
        print('joint')
        print(joint_leds)
        led1_avg = led1_data.groupby(y_name, as_index=True).mean()
        led2_avg = led2_data.groupby(y_name, as_index=True).mean()

        joint_avg = joint_leds.groupby(y_name, as_index=True).mean()

        res = (joint_avg.iloc[0] - led1_avg.iloc[0] - led2_avg.iloc[0])
        print(res.T)
        # print(type(res))
        # res_df = pd.DataFrame(res, index=["{0} {1}".format(current, int_time)])
        res_df = pd.DataFrame(res)  #, columns=["{0} {1}".format(current, int_time)])
        res_df.columns = ["{0} {1}".format(current, int_time)]
        residues = pd.concat([residues, res_df], axis=1, ignore_index=False)
        # residues.append(res.T, ignore_index=False)
        # print(res_df)
        # print(res_df.shape)
        print('-----', current, int_time)
        # print(residues)
        # ham

print(residues)
print(residues.shape)
residues.T.to_csv("as7265x_1A_residues.csv")
