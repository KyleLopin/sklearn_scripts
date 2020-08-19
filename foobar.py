# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import data_getter

# a = np.array([2, 1, 5, 6, 10])
# b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
#
# # print(b)
# # sort_indices = np.argsort(a)
# # print(sort_indices[:-1])
# # print(b[:, sort_indices[:-1]])
# #
# # print(int(90/100))
#
# x = np.zeros([5, 2])
# print(x)
# b = [3, 4, 5]
# x[2:, 0] = b
# print(x)
# c = [10, 20]
# x[2, :] = 99
# print(x)
#
# diagram = Digraph('decision tree', node_attr={'style': 'rounded', 'shape': 'rectangle'})
# diagram.attr(size='6,6')
#
# a1 = u'Leaf Nitrogen <= 1 g m\u207B\u00B2'
# # diagram.node(a1, color='lightblue')
# b1 = "Add fertilizer\nLow plant nitrogen"
# diagram.node(b1, fillcolor='coral1', style='filled')
# diagram.edge(a1, b1, label="True")
# a2 = u'Leaf Nitrogen >= 1.5 g m\u207B\u00B2'
#
# diagram.edge(a1, a2, label="False")
#
# diagram.edge(a2, "Good growing conditions?", label='False')
# b2 = "Don't add fertilizer\nAdequate plant nitrogen"
# diagram.node(b2, fillcolor='lightblue', style='filled')
# diagram.edge(a2, b2, label="True")
# b3 = "Add fertilizer\nto optimize growth"
# diagram.node(b3, fillcolor='coral1', style='filled')
# diagram.edge("Good growing conditions?", b3, label="True")
#
# a3 = "Don't add fertilizer\nNot optimal growing conditions"
# diagram.node(a3, fillcolor='lightblue3', style='filled')
# diagram.edge("Good growing conditions?", a3, label="False")
#
# print(dir(diagram))
# print(diagram.node)
# print(dir(diagram.node))
# print(diagram.node_attr)
# print(dir(diagram.edge))
#
# diagram.view()
# x_data, data = data_getter.get_data('as7263 mango')
# print(data)
# print(x_data)
# data.to_csv("as7263_mango.csv")
# df = pd.DataFrame([
#   [1, '3 inch screw', 0.5, 0.75],
#   [2, '2 inch nail', 0.10, 0.25],
#   [3, 'hammer', 3.00, 5.50],
#   [4, 'screwdriver', 2.50, 3.00]
# ],
#   columns=['Product ID', 'Description', 'Cost to Manufacture', 'Price']
# )
#
# print(df)
#
# df2 = pd.DataFrame([
#   [1, 5, 10],
#   [2, 20, 25],
#   [3, 30, 35],
#   [4, 40, 45]
# ],
#   columns=['C1', 'C2', 'C3']
# )
#
# df = pd.concat([df, df2], axis=1)
# print(df)

# print(40*np.linspace(.4, 1.0, 5))

# a = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
# a = pd.DataFrame()
# print(a)
# b = pd.DataFrame([[10, 20, 30]], columns=["a", "b", "c"])
# print(b)
# c = pd.concat([a, b], axis=0)
# print('====')
# # print(c)
# training = [0.03096, .0299, .0299]
# train_std = [0.0038, .0033, 0.0033]
# test = [0.053, 0.0502, 0.0494]
# test_std = [0.0096, .0078, 0.00814]
# current_levels = ["10", "50", "200"]
# x = np.arange(len(current_levels))
# fig, ax = plt.subplots()
# ax.bar(x, training, width=0.4, yerr=train_std, label="Training set")
# ax.bar(x+0.4, test, width=0.4, yerr=test_std, label="Training set")
# # ax.set_ylim([0.06, 0.11])
# ax.set_xlim(-0.5, 4.2)
# ax.set_title("Error as a function of Integration cycles\n"
#              "AS7265x with Betel leaves")
# ax.legend(loc='lower right')
# ax.set_xticks(x)
# ax.set_xticklabels(current_levels)
# ax.set_ylabel("Mean Absolute Error")
# ax.set_xlabel("Integration cycles")
# plt.show()
# data = pd.read_csv("tomato_C12880.csv")
#
# print(data)
# data.T.plot()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Fraction of reflectance")
# plt.title("Reflectance of Tomatos")
# plt.show()

# data = pd.read_excel("Chlorophyll mango content.xlsx")
# data = data.dropna()
# print(data)
# data['Leaf No.'] = data['Leaf No.'].astype(int)
# print(data)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 500)
# data.to_csv("mango_chloro.csv")
chloro_data = pd.read_csv("mango_chloro.csv", index_col="Leaf number")
print(chloro_data)
new_columns = ["Leaf weight", "Chlorophyll a",
               "Chlorophyll b",	"Total Chlorophyll",
               "Chlorophyll a STD", "Chlorophyll b STD",
               "Total Chlorophyll STD"]
s_data = pd.read_csv("as7262 mango n.csv")
print(s_data)

for new_column in new_columns:
    s_data.insert(len(s_data.columns), new_column, 0.0)


print(s_data)

# print(s_data["Leaf number"])
leaf_n_column = s_data["Leaf number"]
print(leaf_n_column)

def mapper(leaf_num):
    print('ln: ', leaf_num)
    chlorophyll_level = chloro_data.loc[leaf_num]
    print('cl: ')
    print(chlorophyll_level)
print(chloro_data)
print(chloro_data.columns)
# new_leaf_column = leaf_n_column.map(mapper)

for index, row in s_data.iterrows():
    print(row)
    leaf_num = row["Leaf number"]
    for column in new_columns:
        print('oo')
        print(chloro_data[column])
        s_data.loc[leaf_num, column] = chloro_data[column]
    print(s_data.loc[index])
    print("=======")


