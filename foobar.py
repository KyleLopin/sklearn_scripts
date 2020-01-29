# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

from graphviz import Digraph
import numpy as np
import data_getter

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
x_data, data = data_getter.get_data('as7263 mango')
print(data)
print(x_data)
data.to_csv("as7263_mango.csv")
