# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import pandas as pd


# get control data
ctrl_data = pd.read_excel("first_set_raw.xlsx")
print(ctrl_data)
ctrl_data = ctrl_data.loc[ctrl_data["type exp"] == "control"]
print(ctrl_data)
ctrl_data["health"] = 1.0
print(ctrl_data)

dead_leafs = pd.read_excel("dead_leaves_raw.xlsx")

print(dead_leafs.shape)
print(ctrl_data.shape)
dead_leafs["health"] = 0
# print(dead_leafs)

final_df = pd.concat([ctrl_data, dead_leafs])
final_df.to_excel("ctrl_and_dead_raw.xlsx")
