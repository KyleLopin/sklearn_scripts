# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# local files
import get_data

fm.fontManager.addfont('THSarabunNew.ttf')
plt.rcParams['font.family'] = 'TH Sarabun New'
plt.rcParams['xtick.labelsize'] = 20.0
plt.rcParams['ytick.labelsize'] = 20.0
ALPHA = 0.9


def make_daily_average_figure(date):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                                 constrained_layout=True)
        data, wavelengths = get_data.get_data(date)
        print(data.shape)
        colors = ["navy", "turquoise", "darkorange", "magenta"]

        x_columns = []
        data = data[data['variety'] != "กระดาษขาว"]
        print(data.shape)
        # data = data.groupby(["variety", "type exp"], as_index=False).mean(numeric_only=True)
        # data_std = data.groupby(["variety", "type exp"], as_index=False).std(ddof=1, numeric_only=True)

        # print(data_std.shape)
        for column in data.columns:
            if 'nm' in column:
                x_columns.append(column)
        # x_data = data[x_columns]
        # x_data_std = data_std[x_columns]
        axes.set_title(f"Raw Data: {date}", fontsize=18)
        axes.set_ylabel("Reflectance", fontsize=15)
        axes.set_xlabel("Wavelengths (nm)", fontsize=15)
        for color, variety in zip(colors, data['variety'].unique()):
            for ls, exp_type in zip(['solid', 'dashed'], ['control', 'งดน้ำ']):
                data_slice = data.loc[(data["type exp"] == exp_type) &
                                      (data["variety"] == variety)]
                mean = data_slice[x_columns].mean()
                std = data_slice[x_columns].std()
                axes.plot(wavelengths, mean.T,
                          ls=ls, color=color, alpha=ALPHA,
                          label=f"{variety}, {exp_type}")
                axes.fill_between(wavelengths,
                                  (mean-std).T,
                                  (mean+std).T, color=color,
                                  alpha=0.08)

        axes.legend(prop={'size': 14})
        return fig


if __name__ == "__main__":
    set = 1
    if set == 1:
        dates = ["08-06", "08-08", "08-10", "08-12",
                 "08-14", "08-16", "08-18", "08-20",
                 "08-22", "08-24", "08-26", "08-28",
                 "08-30", "09-01", "09-05", "09-07",
                 "09-09", "09-11"]
    elif set == 2:
        dates = ["08-27", "08-29", "08-31",
                 "09-02", "09-04", "09-06", "09-08",
                 "09-10", "09-12", "09-14", "09-16"
                 "09-18"]
    pdf_file = PdfPages('set_1_average_daily.pdf')
    for date in dates:
        fig = make_daily_average_figure(date)
        pdf_file.savefig(fig)
        # plt.show()
    pdf_file.close()
