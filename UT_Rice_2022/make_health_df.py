# Copyright (c) 2022 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import pandas as pd

SET = "first"
SENSOR = "AS7265x"
TYPE = "reflectance"

# get control data
ctrl_data = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
print(ctrl_data)
ctrl_data = ctrl_data.loc[ctrl_data["type exp"] == "control"]
print(ctrl_data)
ctrl_data["health"] = 1.0
print(ctrl_data)

dead_leafs = pd.read_excel(f"dead_leaves_{TYPE}.xlsx")

print(dead_leafs.shape)
print(ctrl_data.shape)
dead_leafs["health"] = 0
# print(dead_leafs)

final_df = pd.concat([ctrl_data, dead_leafs])
final_df.to_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
