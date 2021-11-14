"""
gta_xch4.py

Obtains known estimate of column-averaged dry mixing ratio of methane (XCH4) 
in the Greater Toronto Area.

Author: Shiqi Xu
"""

import os
import statistics as stats
from typing import Tuple

import pandas as pd
from matplotlib import pyplot as plt

def known_gta_xch4(save_fig=False) -> Tuple[float, float]:
    """Generates a histogram of 2018-19 XCH4 measurements from Wunch lab, 
    and calculates mean and standard deviation in ppm.

    Args:
        save_fig (bool, optional): Whether to save plot. Defaults to False.

    Returns:
        mean_xch4 (float): Mean (ppm) calculated from filtered dataset.
        stdev_xch4 (float): Standard deviation (ppm) calculated from filtered dataset.
    """
    path_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    path_data = os.path.join(path_root, "data", "ta_20180601_20190930.oof.csv")

    data = pd.read_csv(path_data, skiprows=226)
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

    mean_xch4 = stats.mean(xch4)
    stdev_xch4 = stats.stdev(xch4)

    print("Mean:", mean_xch4)
    print("Std Dev:", stdev_xch4)

    if save_fig == True:
        path_plot = os.path.join(path_root, "plots", "GTA_XCH4_2018-19_hist.png")
        plt.savefig(path_plot, facecolor="white")
        plt.close()

    plt.show()

    return mean_xch4, stdev_xch4
