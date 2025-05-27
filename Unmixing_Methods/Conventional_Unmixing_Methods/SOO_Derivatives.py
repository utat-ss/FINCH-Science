# Technique: Intervaled Wavelength Range --> Scaled Spectra --> Derivative of Reflectances. Feel free to change the intensity of each.
# Download and replace library (line 86) from your computer.
# When run, this file will produce a comparison of True abundances of green vegetation (GV), dead vegetation (NPV), and Soil from our "library" to optimized 
# abundances found using LMM_FAE.  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =============================================================================
# Dataframe formating: revalues, broad_subset, and subset_index, written by Aidan.
# =============================================================================

def refvalues(df):
    """
    Subset the DataFrame to only include columns that contain reflectance values.
    There are many other column headers in the library that we don't want to use in the analysis because they are all strings.
    We just want wavelength and reflectance values.
    Use df.head() to see the first 5 rows of a df'

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with reflectance columns.
    """
    return df.iloc[:, 7:]


def subset_index(df, indices):
    """
    Subset a DataFrame based on specific row indices.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to subset.
    indices (list or array-like): The specific row indices to keep.
    
    Returns:
    pd.DataFrame: The subsetted DataFrame containing only the rows with the specified indices.
    """
    # Subset the DataFrame
    subset_df = df.loc[indices]
    
    return subset_df


def broad_subset(df, start_idx, end_idx):
    """
    Subset a DataFrame to include columns from start_idx to end_idx. 
    This is for subsetting a continuous range of reflectance values. 
    For FINCH the parameters are 50 and 130

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    start_idx: Starting index of columns to include.
    end_idx: Ending index of columns to include.

    Returns:
    pd.DataFrame: A subsetted DataFrame with selected columns.
    """
    return df.iloc[:, start_idx:end_idx+1]


# =============================================================================
# The functions used for optimization: dataframe_derivative, LMM_FAE 
# =============================================================================

def dataframe_derivative(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ 
    Return DataFrame of the numerical derivation of the contents in each row of all its associated wavelengths
    in each column using np.gradient function.
    """
    dataframe_array = dataframe.values
    dataframe_derivative_array = np.gradient(dataframe_array, axis=1) # Axis=1 means taking deriv across columns
    dataframe_derivative = pd.DataFrame(dataframe_derivative_array, dataframe.index, dataframe.columns)

    return dataframe_derivative


# =============================================================================
# All data and formatting for all Spectra and Endmember (em) Spectra 
# =============================================================================

# Replace the library on your own!! Find on Google Docs... ask Ege.
library = pd.read_csv(r"C:\Users\Zara\OneDrive - University of Toronto\Desktop\UTAT\Data_Files\fractional-cover-simulated-vswir-dataset-version-2--original-10nm-spectra.csv")

true_abundances = library.iloc[:, [2, 3, 5]]

all_spectra = refvalues(library) # And their associated reflectances

em_spectra = subset_index(all_spectra, [47,33,11]) # Randomly chosen combination of endmember indexes [gv, npv, soil], change per test.


# Derivative versions (used for testing only)

em_derivative = dataframe_derivative(em_spectra)

all_derivative = dataframe_derivative(all_spectra)


# Inversion #1: Wavelength Range Intervaled --> Spectra Scaled --> Derivative of Reflectance Taken

intervaled_all_spectra = broad_subset(all_spectra, 110, 124) # Mixed spectra for wavelengths between 1500 and 1640, picked for best result and within FINCH range

intervaled_all_scaled = (intervaled_all_spectra * 100000)

intervaled_em_scaled = subset_index(intervaled_all_scaled, [47,33,11])

intervaled_em_derivative = dataframe_derivative(intervaled_em_scaled)

intervaled_all_derivative = dataframe_derivative(intervaled_all_scaled)


# =============================================================================
# Linear Mixing Model: Fractional abundance estimator
# =============================================================================

def LMM_FAE(spectra: pd.DataFrame, endmembers: pd.DataFrame, plot_results: bool = False):
    """
    Estimate fractional abundances with sum-to-one and greater-than-zero constraints using a linear spectral mixing model.

    Parameters:
    spectra (pd.DataFrame): A DataFrame where each row represents a spectral measurement
                            and each column represents a wavelength or spectral band.
    endmembers (pd.DataFrame): A DataFrame where each row represents an endmember spectrum
                               and each column represents the corresponding wavelength or spectral band.

    Returns:
    A tuple containing:
        - pd.DataFrame: A DataFrame with estimated fractional abundances for each spectrum
        - float: The overall RMSE between the original spectra and the reconstructed spectra
        - pd.Series: A Series containing the RMSE for each individual spectrum
    
    This function performs sequential optimization for each spectrum. 
    """
    def objective(fractions, spectrum, endmembers):
        # Objective function: Mean squared error for a single spectrum
        reconstructed_spectrum = np.dot(fractions, endmembers)
        mse = np.mean((spectrum - reconstructed_spectrum) ** 2)
        return mse

    def constraint_sum_to_one(fractions):
        # Constraint: sum of fractions should be 1
        return np.sum(fractions) - 1
    
    def constraint_greater_than_zero(fractions):
        # Constraint: fractions should be greater than zero
        return fractions - epsilon
    
  
    
    # Convert DataFrames to NumPy arrays for matrix operations
    spectra_array = spectra.values
    endmembers_array = endmembers.values
    num_spectra = spectra_array.shape[0]
    num_endmembers = endmembers_array.shape[0]

    # Epsilon to ensure strictly positive values
    epsilon = 1e-6
    
    # Store the fractional abundances and RMSE for each spectrum
    all_fractions = np.zeros((num_spectra, num_endmembers))
    individual_rmse = []

    # Loop through each spectrum and solve the optimization problem individually
    for i in range(num_spectra):
        spectrum = spectra_array[i, :]
        
        # Initial guess: Equal fractions for the current spectrum
        initial_guess = np.ones(num_endmembers) / num_endmembers
        
        # Constraints for the current spectrum
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                       {'type': 'ineq', 'fun': constraint_greater_than_zero}]
        
        # Optimization for the current spectrum
        result = minimize(objective, initial_guess, args=(spectrum, endmembers_array),
                          constraints=constraints, method='SLSQP', options={'disp': False})
   

        # Extract the optimized fractions for the current spectrum
        fractions = result.x
        all_fractions[i, :] = fractions

        # Reconstruct the spectrum and compute RMSE
        reconstructed_spectrum = np.dot(fractions, endmembers_array)
        mse = np.mean((spectrum - reconstructed_spectrum) ** 2)
        rmse = np.sqrt(mse)
        individual_rmse.append(rmse)
        
        # Optional plotting
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(spectra.columns, spectrum, label='True Spectrum', color='blue')
            plt.plot(spectra.columns, reconstructed_spectrum, label='Reconstructed Spectrum', color='red', linestyle='--')
            plt.title(f'Spectrum {i+1}: True vs Reconstructed')
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.legend()
            plt.xticks(range(0,len(spectra.columns), 10), fontsize=8)
            plt.show()
    
    # Convert the final estimated fractions back to a DataFrame
    fractions_df = pd.DataFrame(all_fractions, index=spectra.index, columns=endmembers.index)
    
    # Create a Series for individual RMSE values
    individual_rmse_series = pd.Series(individual_rmse, index=spectra.index, name='RMSE')

    # Calculate the overall RMSE across all spectra
    overall_rmse = np.mean(individual_rmse)
    return fractions_df, overall_rmse, individual_rmse_series


# =============================================================================
# Formating all optimized abudances from LMM_FAE to compare to real abundances
# =============================================================================

def optimized_abundances(em_spectra: pd.DataFrame, all_spectra: pd.DataFrame):
    """
    Returns one DataFrame of optimized abundances for all spectra, all using LMM_FAE as the abundance optimization method.
    Note: Optimized abundance means that, by analyzing the spectra, LMM_FAE assigns a fractional abundance of green vegetation (GV), 
    non-vegetation/plant material (NVP), and soil components.
    """
    
    optimized_abundances = []
    
    for i in range(all_spectra.shape[0]): 
        
        spectra = subset_index(all_spectra, [i]) # Iterating through each row in dataframe
        
        
        fractions_df, _, _ = LMM_FAE(spectra, em_spectra)         
        optimized_abundances.append(fractions_df)
        
        
    # Making the columns for the DataFrame
    optimized_abundances_df = pd.concat(optimized_abundances, axis=0).reset_index(drop=True)
    optimized_abundances_df.columns = ["gv_fraction", "npv_fraction", "soil_fraction"]
    
    return optimized_abundances_df


# =============================================================================
# Plotting Function
# =============================================================================

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


# If commented out, used for testing/comparison to opt_ab_deriv_intervaled

#opt_ab = optimized_abundances(em_spectra, all_spectra)
#opt_ab_deriv = optimized_abundances(em_derivative, all_derivative) # Not the derivative of optimized abundance, but the spectra used to get it
#opt_ab_intervaled = optimized_abundances(intervaled_em_scaled, intervaled_all_scaled)

#plot_abundance_comparison(true_abundances, opt_ab)
#plot_abundance_comparison(true_abundances, opt_ab_deriv)
#plot_abundance_comparison(true_abundances, opt_ab_intervaled)
#plot_abundance_comparison(true_abundances.iloc[0:600], opt_ab_deriv_intervaled.iloc[0:600])

# =============================================================================
# Called Lines
# =============================================================================

opt_ab_deriv_intervaled = optimized_abundances(intervaled_em_derivative, intervaled_all_derivative)

plot_abundance_comparison(true_abundances, opt_ab_deriv_intervaled)

