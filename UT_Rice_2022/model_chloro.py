# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# standard libraries
import joblib
# installed libraries
import pandas as pd

TYPE = "raw"
SET = "first"
SENSOR = "AS7262"
MODEL = "pls"

data = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
model = joblib.load(f'rice_{SENSOR}_a_{MODEL}.joblib')
x_columns = []
wavelengths = []
for column in data.columns:
    if 'nm' in column:
        x_columns.append(column)
x_data = data[x_columns]
# x_data = x_data.iloc[:, :12]
chloro_predict = model.predict(x_data)
# chloro_predict = x_data.iloc[:, 5] - x_data.iloc[:, 7]
# print(chloro_predict)
data["chloro"] = chloro_predict
data.to_excel(f"chloro_a_{SENSOR}_{SET}_set_{TYPE}_{MODEL}.xlsx")
