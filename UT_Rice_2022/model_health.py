# Copyright (c) 2020 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

# fm.fontManager.addfont('THSarabunNew.ttf')
# plt.rcParams['font.family'] = 'TH Sarabun New'
# plt.rcParams['xtick.labelsize'] = 20.0
# plt.rcParams['ytick.labelsize'] = 20.0
AXIS_LABEL_FONTSIZE = 20
TITLE_LABEL_FONTSIZE = 30

df = pd.read_excel("ctrl_and_dead_raw.xlsx")
print(df)
x_columns = []
for column in df.columns:
    if 'nm' in column:
        x_columns.append(column)

x_data = df[x_columns]
regr = PLSRegression(n_components=6)
regr = SVR()

y = df["health"]

regr.fit(x_data, y)
y_predict = regr.predict(x_data)
print(f"score: {regr.score(x_data, y)}")
plt.scatter(y, y_predict)
plt.show()

all_data = pd.read_excel("first_set_raw.xlsx")
x_data = all_data[x_columns]
modeled_health = regr.predict(x_data)
print(modeled_health)
all_data["modeled_health"] = modeled_health
print(all_data)
all_data.to_excel("modeled_health_first_raw.xlsx")
