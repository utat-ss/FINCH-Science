import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from numba import jit,njit, prange, cuda
import time

#if you want to parallelize or not
parallel_computing = False

#region Compiling and Functions

if parallel_computing == False:
    def mse(fractions, endmembers, spectrum):
    # Objective function: Mean squared error for a single spectrum
        reconstructed_spectrum = np.dot(fractions, endmembers)
        return np.mean((spectrum - reconstructed_spectrum)**2)


    def compute_mses(abundance_matrix, endmembers, spectrum):
        mse_values = np.zeros(abundance_matrix.shape[0])
        for i in prange(abundance_matrix.shape[0]):
            mse_values[i] = mse(abundance_matrix[i], endmembers, spectrum)
        return mse_values

elif parallel_computing == True:

    @jit
    def mse(fractions, endmembers, spectrum):
    # Objective function: Mean squared error for a single spectrum
        reconstructed_spectrum = np.dot(fractions, endmembers)
        return np.mean((spectrum - reconstructed_spectrum)**2)

    @jit(parallel=True)
    def compute_mses(abundance_matrix, endmembers, spectrum):
        mse_values = np.zeros(abundance_matrix.shape[0])
        for i in prange(abundance_matrix.shape[0]):
            mse_values[i] = mse(abundance_matrix[i], endmembers, spectrum)
        return mse_values

startcompile = time.time()

@njit
def create_abundance_space(A_vals):
    abundance_space = []
    
    for i in range(len(A_vals)):
        for j in range(len(A_vals)):
            k = 1.0 - (A_vals[i] + A_vals[j])
            if 0 <= k <= 1:
                abundance_space.append((A_vals[i], A_vals[j], k))
    return np.array(abundance_space)


"""
abundance_combinations = []
for abundances in product(A_vals, repeat=n_endmembers):
    if np.isclose(sum(abundances), 1.0):
        abundance_combinations.append(np.array(abundances))
"""

endcompile = time.time()

print("Compile time is:", (endcompile-startcompile) * (10**3), "ms")

#endregion

#region Load Data

startload = time.time()

start_row = 56
end_row  = 120

# Load endmember data from XLSX file
endmember_file_path = r'C:\University\science utat\endmember_perfect_1.csv'  # Update with actual file path
endmembers = ((pd.read_csv(endmember_file_path, header=None).values)[[2,1,3],start_row:end_row]).astype(np.float64) # Read as numpy array


# Load spectrum data from XLSX file
spectrum_file_path = r'C:\University\science utat\simpler_data.csv'  # Update with actual file path
spectra = ((pd.read_csv(spectrum_file_path, header=None).values)[:,start_row:end_row]).astype(np.float64)  # Read as numpy array


# Specify the row number to process (0-based index)
n = 0   +1  # Example: process the 1st row (index 0), change the left of +1 term
spectrum = spectra[n, :]  # Extract the nth row (spectrum)

n_endmembers = endmembers.shape[0]  # Number of endmembers (should be 3 in this case)

#endregion

num_points = 100

#region Definitions

# Define the search space (brute force search)

print("Search space fineness:", num_points)

A_vals = np.linspace(0, 1, num_points)

#To store MSE values
mse_values = []

#To store possible abundance combinations, ensure they sum up to one

abundance_space = create_abundance_space(A_vals)

abundance_matrix = np.array(abundance_space, dtype=np.float64)

endload = time.time()

print("Load time is:", endload-startload, "sec")

#endregion

#region Compute

startcompute = time.time()

mse_values = np.array(compute_mses(abundance_matrix, endmembers, spectrum))

endcompute = time.time()

if parallel_computing==True:
    print("Compute time with parallelization is:",endcompute-startcompute, "sec")
else:
    print("Compute time without parallelization is:",endcompute-startcompute, "sec")

#endregion

#region Analysis

startanalysis = time.time()

# Find the minimum MSE and corresponding estimated abundances
min_index = np.argmin(mse_values)
min_abundances = abundance_matrix[min_index]

endanalysis = time.time()
print("Analysis time is:",endanalysis-startanalysis, "sec")

#endregion

#region Text Output

print(f"Processing spectrum row {n}:")
print(f"Minimum MSE: {mse_values[min_index]:.6f}")
for i, min_abundance in enumerate(min_abundances):
    print(f"Estimated A{i+1}: {min_abundance:.2f}")

#endregion

#region Plots

# 3D Plot of MSE with A1 and A2 as the x and y axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')


# Scatter plot for all points
scatter = ax.scatter(
    abundance_matrix[:,0],  # A1
    abundance_matrix[:,1],  # A2
    mse_values,          # MSE
    c=mse_values,        # Color by MSE
    cmap='coolwarm',
    marker='o'
)

# Highlight the point with the minimum MSE
ax.scatter(
    min_abundances[0],   # A1
    min_abundances[1],   # A2
    mse_values[min_index],  # MSE
    color='black',
    s=100,
    label='Minimum MSE'
)

# Add labels and title
ax.set_xlabel('Estimated A1')
ax.set_ylabel('Estimated A2')
ax.set_zlabel('MSE')
ax.set_title(f'Brute Force MSE Estimation for Spectrum Row {n} (3 Endmembers)')
plt.legend()
plt.colorbar(scatter, label='MSE')
plt.show()

endplots = time.time()

print("Plot time is:",endanalysis-startanalysis, "sec")

closest_reconstructed = np.dot(min_abundances, endmembers)
wavelength_space = spectra[0, :]
plt.plot(wavelength_space, closest_reconstructed, color='black', label = 'Reconstructed')
plt.plot(wavelength_space, spectrum, color= "green", label="Actual")
plt.legend()
plt.show()

#endregion
