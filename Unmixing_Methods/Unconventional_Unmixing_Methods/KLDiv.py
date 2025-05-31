"""
This code uses the Kullback-Leibler Divergence (KL Divergence) to unmix.
It really is completely unconventional and weird.

For some theory, ask Ege.
"""

import numpy as np
import pandas as pd
from scipy.special import kl_div
import matplotlib.pyplot as plt

def kl_divergence_unmixing(spectrum, endmembers):

    """
    Parameters:

    spectrum: numpy array of the spectrum to unnmix of shape (1, wavelength#)
    endmembers: numpy array of the endmembers to unmix of shape (3, wavelength#) where the rows are gv, npv, soil in order

    Retuurns: an array (1,3) of predicted abundances for gv, npv, soil in order
    """

    # We first have to ensure the spectra and EMs are normalized since KL divergence requires a probability distribution-like input

    spectrum = spectrum / np.sum(spectrum)
    endmembers = endmembers / np.sum(endmembers, axis=1, keepdims=True)

    # Now we proceed with calculating the KL divergence for each EM

    kl_divergences = np.array([kl_div(spectrum, endmember) for endmember in endmembers])

    """
    This bit is a bit long. KL divergence is =0 when two inputs are the same, so the smaller KL Div, the more it can explain the spectrum.
    Therefore, we take the inverse of the KL Div to get a more intuitive measure of how well each endmember explains the spectrum; in this 
    case the higher the value, the better it explains the spectrum. After that, we obviously need to normalize the values to get 'abundances'.
    """

    kl_divergences = 1 / (kl_divergences + 1e-4) # small constant added to avoid division by zero
    abundances = kl_divergences / np.sum(kl_divergences)

    return abundances

def plot_abundance_comparison(true_ab_df: pd.DataFrame, optimized_ab_df: pd.DataFrame, title: str = "Abundance Comparison"):
    """
    Creates a scatter plot comparing optimized abundance (y-axis) with true abundances (x-axis).
    """
    
    # Create a scatter plot for each column (abundance type)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for column in optimized_ab_df.columns:
        ax.scatter(optimized_ab_df[column], true_ab_df[column], label=column)
    
    ax.set_xlabel('Optimized Abundance')
    ax.set_ylabel('True Abundance')
    ax.set_title(title)
    ax.legend()

    plt.show()

