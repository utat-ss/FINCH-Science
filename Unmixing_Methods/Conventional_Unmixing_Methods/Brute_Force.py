import numpy as np
import pandas as pd
from itertools import product
from numba import njit, prange
import matplotlib.pyplot as plt 
import time

# Function to generate the index matrix for endmember combinations 

def generate_combinations(endmember_file_handle: str, endmember_limit: float):

    """Generates the endmember combination space

    Parameters
        endmember_file_handle (str): The path of file where endmembers will be collected from.
        endmember_limit (float in [0.5,1]): The float number above which a spectrum with abundance is defined as an endmember below 0.6 uses numba as it is faster in that range.
        

    Returns
        index_matrix (np.array): Matrix where each row represents a combination of EM indices.
        
    """

    # Load endmember data from CSV file
    df = pd.read_csv(endmember_file_handle)

    # Filter indices based on endmember limit
    gv_indices = df.index[df['gv_fraction'] >= endmember_limit].to_list()
    npv_indices = df.index[df['npv_fraction'] >= endmember_limit].to_list()
    soil_indices = df.index[df['soil_fraction'] >= endmember_limit].to_list()

    # Generate all possible combinations of EM indices 
    index_matrix = np.array(list(product(gv_indices, npv_indices, soil_indices)), dtype=np.int32)
    return index_matrix

#Calculates the mse using numba 
@njit
def mse(fractions, endmembers, spectrum):
    """ Compute mean squared error

    Parameters
        fractions (np.array): Estimated abundances of 3 endmembers 
        endmembers (np.array): Numpy array consisting of endmember spectra 
        spectrum (np.array): Actual spectra extracted from the data file 

    Returns
        mse (float): Mean squared error between actual and reconstructed spectrum 
        
      """
    reconstructed_spectrum = np.dot(fractions, endmembers)
    return np.mean((spectrum - reconstructed_spectrum) ** 2)

#Search through the abundance space and find the mse values for each abundance combination
@njit
def compute_mses(abundance_matrix, endmembers, spectrum):
    """ Search through all possible abundances in abundance space matrix and compute MSE for each combination. 

    Parameters 
        abundance_matrix (np.array): The matrix consists of all possible abundance combinations
        endmembers (np.array): Numpy array consisting of endmember spectra
        spectrum (np.array): Actual spectra extracted from the data file

    Returns 
        mse_values (np.array): Array with MSEs stored for each combination of abundances in the abundance space 
     
     """
    mse_values = np.zeros((abundance_matrix.shape[0]))
    for i in prange(abundance_matrix.shape[0]):  
        mse_values[i] = mse(abundance_matrix[i], endmembers, spectrum)
    return mse_values

#Creates the abundance space that the search will be done on 
@njit
def create_abundance_space(A_vals):
    """ Create the abundance space of all possible unique abundance combinations, ensuring a "sum-to-one" constraint.
     
    Parameters 
        A_vals (np.array): Numpy array consisting of abundance values given a specific space fineness
         
    Returns
        abundance_matrix (np.array): The matrix consists of all possible abundance combinations
    
     """
    abundance_matrix = []
    for i in range(len(A_vals)):
        for j in range(len(A_vals)):
            k = 1.0 - (A_vals[i] + A_vals[j])
            if 0 <= k <= 1:
                abundance_matrix.append((A_vals[i], A_vals[j], k))
    return np.array(abundance_matrix)

@njit
def brute_force(endmembers, spectrum, abundance_matrix):
    """ Brute force search for best abundance combination. Searches with a given combination of endmembers on spectrum from a dataset.
    Calculates minimum MSE and the corresponding abundances.
    
    Parameters
        endmembers (np.array): Numpy array consisting of endmember spectra
        spectrum (np.array): Actual spectra extracted from the data file
        abundance_matrix (np.array): The matrix consists of all possible abundance combinations
        
    Returns
        min_mse (float): Minimum calculated MSE
        min_abundances (np.array): Abundances that reconstuct the spectrum to give min_mse
        
    """

    # Compute MSE for each combination
    mse_values = compute_mses(abundance_matrix, endmembers, spectrum)

    # Find best combination
    min_index = np.argmin(mse_values)
    min_mse = mse_values[min_index]
    min_abundances = abundance_matrix[min_index]

    return min_mse, min_abundances

@njit(parallel=True)
def compute_best_combination(index_matrix, endmembers_data, spectra, space_fineness =100,  num_rows=50):
    """ Compute the best endmember combination for multiple rows 
    
    Parameters
        index_matrix (np.array): Matrix with indices of all possible EM combinations 
        endmember_data (np.array): Dataset with all EM spectra 
        spectra (np.array): Dataset with the measured spectra 
        space_fineness (int): Number of possible abundance values in range [0,1]
        num_rows (int): Number of rows from the dataset spectra that needs to be processed 

    Returns
        best_avg_mse(np.array): The best average MSE across all the rows
        best_EM (np.array): Best EM combination across all the rows  

    """

    # Generate abundance combinations
    num_combinations = index_matrix.shape[0]
    A_vals = np.linspace(0, 1, space_fineness)
    abundance_matrix = create_abundance_space(A_vals)
    minimum_mses = np.zeros((num_rows, num_combinations))  # Store MSE for each row-combination pair
    predicted_abundances = [] # Store best predicted abundances for all rows

    for row_index in prange(num_rows):  # Iterate through 50 rows (default)
        spectrum = spectra[row_index]
        minimum_abundances = [] # Reset for each row to save memory 
        for i in prange(num_combinations):  # Parallel processing for combinations
            endmembers = endmembers_data[index_matrix[i]]
            min_mse, min_abundances  = brute_force(endmembers, spectrum,abundance_matrix)
            minimum_mses[row_index, i] = min_mse  # Store MSE
            minimum_abundances.append(min_abundances)
        best_index = np.argmin(minimum_mses[row_index]) # Find the abundance combination with the lowest MSE for the current row
        predicted_abundances.append(minimum_abundances[best_index]) # Store the best predicted abundances for the current row
        
           

    # Compute the average MSE per combination across all 50 rows (manually)
    avg_mses = np.zeros(num_combinations)
    for i in prange(num_combinations):  # Column-wise sum
        avg_mses[i] = np.sum(minimum_mses[:, i]) / num_rows

    # Find the best combination (lowest average MSE)
    best_index = np.argmin(avg_mses)

    best_avg_mse = avg_mses[best_index]
    best_EM = index_matrix[best_index]
    
    return best_avg_mse, best_EM, predicted_abundances


# Region Plot
def plot_abundances(predicted_abundances, true_abundances,number_of_rows):
    # Plot estimated vs actual abundances

    true_A1 = true_abundances[:number_of_rows, 0]
    true_A2 = true_abundances[:number_of_rows, 1] 
    true_A3 = true_abundances[:number_of_rows, 2]

    estimated_A1 = predicted_abundances[:, 0]
    estimated_A2 = predicted_abundances[:, 1]
    estimated_A3 = predicted_abundances[:, 2]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(true_A1, estimated_A1, c='r', label='A1')
    plt.xlabel('Actual A1')
    plt.ylabel('Estimated A1')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.subplot(1, 3, 2)
    plt.scatter(true_A2, estimated_A2, c='g', label='A2')
    plt.xlabel('Actual A2')
    plt.ylabel('Estimated A2')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.subplot(1, 3, 3)
    plt.scatter(true_A3, estimated_A3, c='b', label='A3')
    plt.xlabel('Actual A3')
    plt.ylabel('Estimated A3')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.tight_layout()
    plt.show()




# Main execution
if __name__ == "__main__":
    start_time = time.time()

    endmember_file_path = r'C:\Users\zahar\Desktop\FINCH-Science\THE PIPELINE\endmember_perfect_1.csv'
    spectrum_file_path = r'C:\Users\zahar\Desktop\FINCH-Science\THE PIPELINE\simpler_data.csv'

    # Load index matrix
    index_matrix = generate_combinations(endmember_file_path, 1)

    # Load dataset (assuming CSV is stored locally)
    endmembers_data = pd.read_csv(endmember_file_path).values[:, 56:120].astype(np.float64)
    spectra = pd.read_csv(spectrum_file_path).values[:, 56:120].astype(np.float64)
    true_abundances = pd.read_csv(spectrum_file_path).values[:, 1:4].astype(np.float64)

    # Compute best combination across n rows with specified space fineness
    space_fineness = 100
    number_of_rows = 100
    best_avg_mse, best_combination, predicted_abundances = compute_best_combination(index_matrix, endmembers_data, spectra,space_fineness, number_of_rows)
    predicted_abundances = np.array(predicted_abundances)
    
    # Plot estimated vs actual abundances
    plot_abundances(predicted_abundances, true_abundances,number_of_rows)
    
    print(f"Best average MSE: {best_avg_mse}")
    print(f"Best EM combination (on average): {best_combination}")

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds")
