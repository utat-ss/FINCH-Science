"""
This code uses the Kullback-Leibler Divergence (KL Divergence) to unmix.
It really is completely unconventional and weird.

For some theory, ask Ege.
"""

import numpy as np
import pandas as pd
from scipy.special import rel_entr, softmax
import matplotlib.pyplot as plt

def kl_divergence_unmixing(spectrum: np.array, endmembers: np.array, softmax_option: bool = False) -> np.array:

    """
    Parameters:

    spectrum: numpy array of the spectrum to unnmix of shape (1, wavelength#)
    endmembers: numpy array of the endmembers to unmix of shape (3, wavelength#) where the rows are gv, npv, soil in order

    Returns: an array (1,3) of predicted abundances for gv, npv, soil in order
    """

    # We first have to ensure the spectra and EMs are normalized since KL divergence requires a probability distribution-like input

    spectrum = spectrum / np.sum(spectrum)
    endmembers = endmembers / np.sum(endmembers, axis=1, keepdims=True)

    # Now we proceed with calculating the KL divergence for each EM

    kl_divergences = np.array([sum(rel_entr(spectrum, endmember)) for endmember in endmembers])
    
    """
    This part is a bit long. KL divergence is =0 when two inputs are the same, so the smaller KL Div, the more it can explain the spectrum.
    Therefore, we take the inverse of the KL Div to get a more intuitive measure of how well each endmember explains the spectrum; in this 
    case the higher the value, the better it explains the spectrum. After that, we obviously need to normalize the values to get 'abundances'.
    """

    if softmax_option:
        kl_divergences = softmax(kl_divergences)
        abundances = kl_divergences / np.sum(kl_divergences)
    else:
        kl_divergences = 1 / (kl_divergences + 1e-4) # small constant added to avoid division by zero
        abundances = kl_divergences / np.sum(kl_divergences)

    return abundances

def plot_abundance_comparison(true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame, title: str = "Abundance Comparison"):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    """

    types = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    colors = ['green', 'blue', 'brown']
    
    # Create a scatter plot for each column (abundance type)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for column in optimized_ab_df.columns:
        ax.scatter(optimized_ab_df[column], true_ab_df[column], label=types[column], color=colors[column])
    
    ax.set_xlabel('Optimized Abundance')
    ax.set_ylabel('True Abundance')
    ax.set_title(title)
    ax.legend()

    plt.show()

def plot_single_abundance_comparison(abundance_type: int, true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame, title: str = "Abundance Comparison"):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    """

    types = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    colors = ['green', 'blue', 'brown']
    
    # Create a scatter plot for each column (abundance type)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(optimized_ab_df[abundance_type], true_ab_df[abundance_type], label=types[abundance_type], color=colors[abundance_type])
    
    ax.set_xlabel('Optimized Abundance')
    ax.set_ylabel('True Abundance')
    ax.set_title(title)
    ax.legend()

    plt.show()

