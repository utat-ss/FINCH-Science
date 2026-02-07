import pandas as pd
import numpy as np
import pathlib
import re
import scipy
from sklearn import linear_model
import matplotlib.pyplot as plt

"""

My personal utils for working with the spectra data.

Uses all operations are using Matplotlib, Scikit learn, numpy, and pandas

Author: Andrei Akopian

"""

# ----
# Operations on Dataframes
# ----

def open_file(filename):
    path = pathlib.PurePath(filename)
    file_format = path.suffix
    parsing_functions = {
        ".csv" : pd.read_csv,
    }
    return parsing_functions[file_format](filename)

def take_subset(df,start,end):
    """ grab a subset of wavelengths from the dataframe

    return (npv_fractions, spectra, spectra_sources)
    """

    columns = df.columns.to_list()
    wanted = []
    for c in columns:
        if c.isdigit():
            if start<=int(c)<=end:
                wanted.append(c)
    fractions = df[["npv_fraction","gv_fraction","soil_fraction"]]
    spectra = df[wanted]
    spectra_sources = df[["Spectra"]]
    return fractions, spectra, spectra_sources

def split_by_dataset(data,npv_sorted=True):
    """
    Split dataset by the study they came from.
    Also split each subset by npv_fraction
    """
    names = list(data["Spectra"])
    def name_swapping(name):
        return re.sub(r"[\d|\s|\_](.+)$","",name)
    names = list(map(name_swapping,names))

    row_i = 0
    d = dict()
    for n in names:
        if n in d:
            d[n].append(row_i)
        else:
            d[n]=[row_i]
        row_i+=1

    for k in d:
        d[k] = find_runs(d[k])

    frames = []
    for n in d:
        frame = pd.DataFrame()
        for run in d[n]:
            frame = pd.concat([frame, data.iloc[run[0]:run[0]+run[1]]])
        if npv_sorted:
            frame.sort_values(by="npv_fraction")
        frames.append(frame)

    print("Number of Datasets:",len(d))
    return frames, list(d.keys())

def find_runs(values):
    """
    Helper function for split_by_dataset function
    Convert lists of values into (start,length) tuples.
    """
    if not values:
        return []

    runs = []
    start = values[0]
    length = 1

    for i in range(1, len(values)):
        if values[i] == values[i - 1] + 1:
            length += 1
        else:
            runs.append((start, length))
            start = values[i]
            length = 1

    runs.append((start, length))
    return runs

def wavelength_columns(df):
    """
    Get columns headers correspending to wavelengths, based on the fact that the columns headers are numeric
    """
    columns = df.columns.to_list()
    wavelengths = [c for c in columns if c.isdigit()]
    return wavelengths
    
# ----
# Analysis
# ----

def train_linear_model(train_X,train_y):
    reg = linear_model.LinearRegression()
    reg.fit(X=train_X,y=train_y)
    print("Training R^2:",round(reg.score(train_X,train_y),4))
    return reg

def validate(model,train_X,validate_X,train_y,validate_y):
    print("Training R^2:",round(model.score(train_X,train_y),4))
    print("Validation R^2:",round(model.score(validate_X,validate_y),4))

def evaluate_model(predictions,actual):
    """First value is r^2"""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(actual, predictions)
    return r_value**2, (slope, intercept, r_value, p_value, std_err)

def plot_matrix(matrix,title="Distances",colorbar_label="colorbar",x_labels=None,y_labels=None, xlabel="Samples", ylabel="Samples"):
    plt.style.use('_mpl-gallery-nogrid')

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)), labels=x_labels,
              rotation=45, ha="right", rotation_mode="anchor")
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)), labels=y_labels)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    im = ax.imshow(matrix, origin='lower')
    fig.colorbar(im, ax=ax, label=colorbar_label,fraction=0.046, pad=0.04)

    plt.show()

def simple_histogram(data=[1,2,3],title="Title",x="x-axis",y='y-axis',bins=10):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.hist(data,bins=bins)
    print()

def simple_dimension_reducer(np_array, output_n):
    """
    """
    diffs = np.diff(np_array)
    instability = np.diff(diffs)
    return 