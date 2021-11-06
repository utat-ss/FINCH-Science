"""
gta_xch4.py

Obtains estimates of column-averaged dry mixing ratio (XCH4) in the Greater Toronto Area - see README

Author: Shiqi Xu

Created on 2021-11-06
"""

import pandas as pd
from matplotlib import pyplot as plt
import statistics as stats


if __name__ == "__main__":

    data = pd.read_csv("ta_20180601_20190930.oof.csv", skiprows=226)
    print(data.shape)

    data_filtered = data[data['flag']==0]
    print(data_filtered.shape)

    xch4 = data_filtered["xch4(ppm)"]
    xch4_error = data_filtered["xch4(ppm)_error"]

    ppm_hist = plt.figure()
    plt.hist(xch4, bins=100)

    plt.xlabel("XCH$_4$ (ppm)")
    plt.ylabel("Count")
    plt.title("Histogram of 2018-19 GTA XCH$_4$")

    mean_str = "Mean: " + str(round(stats.mean(xch4), 3))
    stdev_str = "Std Dev: " + str(round(stats.stdev(xch4), 3))
    plt.gcf().text(0.72, 0.83, mean_str)
    plt.gcf().text(0.72, 0.80, stdev_str)

    print("Mean:", stats.mean(xch4))
    print("Std Dev:", stats.stdev(xch4))

    plt.savefig("GTA_XCH4_2018-19_hist.png", facecolor="white")
    plt.show()
