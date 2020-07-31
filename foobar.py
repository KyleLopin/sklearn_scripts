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
# print(c)
training = [.0776, .0806, 0.08997, 0.09457]
test = [0.07825, 0.0921, 0.0962, 0.102]
current_levels = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
x = np.arange(len(current_levels))
fig, ax = plt.subplots()
ax.bar(x, training, width=0.4, label="Training set")
ax.bar(x+0.4, test, width=0.4, label="Training set")
ax.set_ylim([0.06, 0.11])
ax.set_xlim(-0.5, 4.5)
ax.set_title("Error as a function of LED current\n"
             "AS7262 with Cananga leaves")
ax.legend(loc='lower right')
ax.set_xticks(x)
ax.set_xticklabels(current_levels)
ax.set_ylabel("Mean Absolute Error")
plt.show()
