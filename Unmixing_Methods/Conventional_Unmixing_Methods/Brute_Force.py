import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time

def mse(estimated_abundances, endmembers, spectrum):
    reconstructed_spectrum = np.dot(estimated_abundances, endmembers)
    return np.mean((spectrum - reconstructed_spectrum) ** 2)

# Start timing
start_time = time.time()

# Load endmember data from CSV file
endmember_file_path = r'C:\Users\zahar\Desktop\FINCH-Science\THE PIPELINE\endmember_perfect_1.csv'
endmember_df = pd.read_csv(endmember_file_path)

# Group data based on fractions
gv_group = endmember_df[endmember_df['gv_fraction'] == 1]
npv_group = endmember_df[endmember_df['npv_fraction'] == 1]
soil_group = endmember_df[endmember_df['soil_fraction'] == 1]

# Select a random row from each group
gv_sample = gv_group.sample(n=1)
npv_sample = npv_group.sample(n=1)
soil_sample = soil_group.sample(n=1)

# Combine selected rows into a single DataFrame and extract numeric data
selected_endmembers = pd.concat([gv_sample, npv_sample, soil_sample]).iloc[:, 5:]
endmembers = selected_endmembers.values  # Convert to numpy array

# Load spectrum data from CSV file
spectrum_file_path = r'C:\Users\zahar\Desktop\FINCH-Science\THE PIPELINE\simpler_data.csv'
spectrum_df = pd.read_csv(spectrum_file_path)

n_endmembers = endmembers.shape[0]  # Number of endmembers (should be 3 in this case)

# Define the search space (brute force search)
num_points = 50
A_vals = np.linspace(0, 1, num_points)

actual_A1 = []
actual_A2 = []
actual_A3 = []
estimated_A1 = []
estimated_A2 = []
estimated_A3 = []

# Select n random rows from the spectrum data
random_rows = spectrum_df.sample(n=300)

# Iterate through the random 300 rows in the spectrum data
for index, row in random_rows.iterrows():
    spectrum = row.iloc[5:].values.flatten()  # Skip first 5 columns
    actual_A1.append(row.iloc[1])  # Extract actual A1 from column 2
    actual_A2.append(row.iloc[2])  # Extract actual A2 from column 3
    actual_A3.append(row.iloc[3])  # Extract actual A3 from column 4
    
    mse_values = []
    A1_list = []
    A2_list = []
    A3_list = []
    
    # Generate all possible abundance combinations ensuring they sum to 1
    for abundances in product(A_vals, repeat=n_endmembers):
        if np.isclose(sum(abundances), 1.0):
            estimated_abundances = np.array(abundances)
            mse_values.append(mse(estimated_abundances, endmembers, spectrum))
            A1_list.append(abundances[0])
            A2_list.append(abundances[1])
            A3_list.append(abundances[2])
    
    mse_values = np.array(mse_values)
    A1_list = np.array(A1_list)
    A2_list = np.array(A2_list)
    A3_list = np.array(A3_list)
    
    # Find the minimum MSE and corresponding estimated abundances
    min_index = np.argmin(mse_values)
    min_A1 = A1_list[min_index]
    min_A2 = A2_list[min_index]
    min_A3 = A3_list[min_index]
    min_mse = mse_values[min_index]
    
    estimated_A1.append(min_A1)
    estimated_A2.append(min_A2)
    estimated_A3.append(min_A3)
    
    

# End timing
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.4f} seconds")

# Plot estimated vs actual abundances
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.scatter(actual_A1, estimated_A1, c='r', label='A1')
plt.xlabel('Actual A1')
plt.ylabel('Estimated A1')
plt.plot([0, 1], [0, 1], 'k--')

plt.subplot(1, 3, 2)
plt.scatter(actual_A2, estimated_A2, c='g', label='A2')
plt.xlabel('Actual A2')
plt.ylabel('Estimated A2')
plt.plot([0, 1], [0, 1], 'k--')

plt.subplot(1, 3, 3)
plt.scatter(actual_A3, estimated_A3, c='b', label='A3')
plt.xlabel('Actual A3')
plt.ylabel('Estimated A3')
plt.plot([0, 1], [0, 1], 'k--')

plt.tight_layout()
plt.show()
