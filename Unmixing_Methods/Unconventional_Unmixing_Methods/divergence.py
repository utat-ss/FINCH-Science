"""
This code uses the Kullback-Leibler Divergence (KL Divergence) to unmix.
It really is completely unconventional and weird.

For some theory, ask Ege.
"""

import numpy as np
import pandas as pd
from scipy.special import rel_entr, kl_div, softmax
import matplotlib.pyplot as plt

def spectrum_to_pdf(spectrum: np.array, wavelengths: np.array) -> np.array:
    
    """
    Parameters:

    spectrum: numpy array (matrix)of the spectrum to turn into PDF of shape (sample_amount, wavelength#)
    wavelengths: numpy array (vector) of the wavelengths corresponding to the spectrum of shape (wavelength#)

    Returns: a numpy array of the PDF of the spectra, normalized to sum to 1, of shape (sample_amount, wavelength#)
    """

    assert (len(spectrum.shape) == 2 and len(wavelengths.shape) == 1), "Spectrum must be a 2D array and wavelengths must be a 1D array."
    assert (spectrum.shape[1]) == wavelengths.shape[0], "Both np arrays must have the same number of wavelengths."

    wavelength_scaled_spectrum = spectrum * wavelengths # Scale points (i,j) by the wavelength (j)
    row_sums = wavelength_scaled_spectrum.sum(axis=1, keepdims=True) # Sum along each j to get total intensity
    pdf = wavelength_scaled_spectrum / row_sums # Normalize with the total intensity on each j

    # Delete intermediary variables to save memory
    del wavelength_scaled_spectrum, row_sums

    return pdf


def divergence_unmixing(spectrum: np.array, endmembers: np.array, wavelengths: np.array = None, compute_option: str = 'Itakura-Saito', output_option: str = 'regular') -> np.array:

    """
    Parameters:

    spectrum: numpy array of the spectrum to unnmix of shape (1, wavelength#)
    endmembers: numpy array of the endmembers to unmix of shape (3, wavelength#) where the rows are gv, npv, soil in order
    wavelengths: if asking for a PDF converting mode, a numpy array of the wavelengths corresponding to the spectrum of shape (wavelength#). Enter nothing if not using such compute mode.
    compute_option: string, either 'Itakura-Saito' or 'Kullback-Leibler'. If 'Itakura-Saito', the Itakura-Saito divergence will be used; if 'Itakura-Saito-PDF', Itakura-Saito div will be used but with PDF initialization; if 'Kullback-Leibler', the Kullback-Leibler divergence will be used; 
    output_option: string, either 'regular' or 'softmax'. If 'regular', the output will be normalized values, if 'softmax', the output will be softmaxed, if 'raw', the output will be raw point-wise KL divergence values

    Returns: an array (1,3) of predicted abundances for gv, npv, soil in order
    """

    # Calculating the point-wise KL divergence for each EM

    if compute_option == 'Itakura-Saito':
        # Weirdly enough, the Itakura-Saito divergence is a generalized form of the KL divergence (it is also known as Bregman divergence) that is implemented in scipy as kl_div
        # Calculate the point-wise IS-Div
        point_wise_div = kl_div(spectrum, endmembers)

    elif compute_option == 'Itakura-Saito-PDF':
        # Itakura-Saito with PDF initialization
        if wavelengths is None:
            raise ValueError("Wavelengths must be provided for the Itakura-Saito-PDF compute option.")
        else:
            # Convert the spectrum and endmembers to PDFs
            pdf_spectrum = spectrum_to_pdf(spectrum, wavelengths)
            pdf_endmembers = spectrum_to_pdf(endmembers, wavelengths)

            # Calculate the point-wise IS-Div
            point_wise_div = kl_div(pdf_spectrum, pdf_endmembers)

            # Delete to save memory
            del pdf_spectrum, pdf_endmembers

    elif compute_option == 'Kullback-Leibler':
        # Regular Kullback-Leibler divergence, needs PDF initialization
        if wavelengths is None:
            raise ValueError("Wavelengths must be provided for the Itakura-Saito-PDF compute option.")
        else:
            # Convert the spectrum and endmembers to PDFs
            pdf_spectrum = spectrum_to_pdf(spectrum, wavelengths)
            pdf_endmembers = spectrum_to_pdf(endmembers, wavelengths)

            # Calculate the point-wise KL-Div
            point_wise_div = rel_entr(pdf_spectrum, pdf_endmembers)

            # Delete to save memory
            del pdf_spectrum, pdf_endmembers

    """
    This part is a bit long. KL divergence is =0 when two inputs are the same, so the smaller KL Div, the more it can explain the spectrum.
    Therefore, we take the inverse of the KL Div to get a more intuitive measure of how well each endmember explains the spectrum; in this 
    case the higher the value, the better it explains the spectrum. After that, we obviously need to normalize the values to get 'abundances'.
    """

    if output_option == 'regular':
        total_div = point_wise_div.sum(axis=1) # Sum across point-wise divs for each column, get (3,)
        del point_wise_div # Delete to save memory

        return (1 / (total_div + 1e-4)) / np.sum(1 / (total_div + 1e-4))  # Invert so that the lower the KL-Div, the higher the abundance, and normalize to sum to 1

    elif output_option == 'softmax':
        total_div = point_wise_div.sum(axis=1) # Sum across point-wise divs for each column, get (3,)
        del point_wise_div # Delete to save memory
        
        return softmax(1 / (total_div + 1e-4)) # Invert so that softmax makes sense and apply softmax, to get probabilities for each model

    elif output_option == 'raw':
        # Return the straight up point-wise KL-Div or IS-Div values
        return point_wise_div
    

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

