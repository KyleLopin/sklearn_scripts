# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
pd.set_option('display.max_columns', 500)
# fm.fontManager.addfont('THSarabunNew.ttf')
# plt.rcParams['font.family'] = 'TH Sarabun New'
# plt.rcParams['xtick.labelsize'] = 20.0
# plt.rcParams['ytick.labelsize'] = 20.0
AXIS_LABEL_FONTSIZE = 20
TITLE_LABEL_FONTSIZE = 30
SET = "first"
SENSOR = "AS7262"
TYPE = "reflectance"

# dead_leaf_data = pd.read_excel(f"ctrl_and_dead_{SET}_{SENSOR}_{TYPE}.xlsx")
dead_leaf_data = pd.read_excel(f"dead_leaves_{SENSOR}_{TYPE}.xlsx")
# dead_leaf_data = dead_leaf_data.loc[dead_leaf_data["health"] == 0]
dead_leaf_data.loc[:, 'health'] = 0
all_data = pd.read_excel(f"{SET}_set_{SENSOR}_{TYPE}.xlsx")
print(dead_leaf_data)
VARIETIES = ['กข43', 'กข79', 'กข85', 'ปทุมธานี 1']
print(VARIETIES)


x_columns = []
for column in dead_leaf_data.columns:
    if 'nm' in column:
        x_columns.append(column)

x_data = dead_leaf_data[x_columns]
regr = PLSRegression(n_components=3)
regr = SVR()

# add each day to the model data
days = all_data["day"].unique()
print(days)
# days = [12, 14, 16, 18, 20, 22, 24, 26,]
all_data["modeled_health"] = None
for day in days:
    new_data = all_data.loc[all_data["day"] == day]
    new_ctrl_data = new_data.loc[new_data["type exp"] == "control"].copy()
    new_ctrl_data.loc[:, "health"] = 1
    # print(new_ctrl_data)
    daily_df = pd.concat([dead_leaf_data, new_ctrl_data], ignore_index=True)

    daily_df = daily_df.loc[daily_df["variety"] != "กระดาษขาว"]
    print(daily_df["health"])
    print(daily_df)
    regr.fit(daily_df[x_columns], daily_df["health"])
    y_predict = regr.predict(daily_df[x_columns])
    plt.scatter(daily_df["health"], y_predict)
    print(daily_df['day'])
    print(regr.score(daily_df[x_columns], daily_df["health"]))

    health_predict = regr.predict(new_data[x_columns])
    all_data.loc[all_data["day"] == day, "modeled_health"] = health_predict

    plt.show()
all_data.to_excel(f"daily_modeled_health_{SET}_{SENSOR}_{TYPE}.xlsx")

y = all_data["modeled_health"]

# regr.fit(x_data, y)
# y_predict = regr.predict(x_data)
# print(f"score: {regr.score(x_data, y)}")
# plt.scatter(y, y_predict)
plt.show()

x_data = all_data[x_columns]
modeled_health = regr.predict(x_data)
print(modeled_health)
all_data["modeled_health"] = modeled_health
print(all_data)
# all_data.to_excel(f"modeled_health_{SET}_{SENSOR}_{TYPE}.xlsx")
