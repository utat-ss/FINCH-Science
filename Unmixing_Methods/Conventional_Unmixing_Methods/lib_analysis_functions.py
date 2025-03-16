# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:26:12 2024

@author: Aidan
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from itertools import combinations
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def endmember(df, col_endmember, threshold=0.90):
    '''
    I define an endmember as a reference within the library that consists of over n% of one material.

    The spectral library has % abundance information for each reference.

    Parameters:
        df (pd.DataFrame): The input DataFrame
        col_endmemebr: the column in df that contains abundance info for a material
        threshold: the sorting percentage

    Returns:
        pd.DataFrame: A DataFrame with  only endmembers of a certain material
    
    Adjust Threshold for different endmember definitions
    '''
    return df[df[col_endmember] > threshold]


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


def plot_endmember(df, row_index, title):
    """
    Plot an endmember

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    row_index: The index of the row to plot.
    title: the name yo
    """
    # Extract the specified row
    row = df.iloc[row_index]
    
    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.plot(row.index, row.values, linestyle='-', marker='o')
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.title(f' {title}, Entry {row_index}')
    plt.grid(True)
    plt.show()


def resample_reflectance_df(df, new_interval, method='linear'):
    """
    Resample reflectance values at new intervals from a DataFrame.
    This allows for different resampling methods to increase or decrease the bandwidth.
    Use before or after subsetting.
    
    Parameters:
        df (pd.DataFrame):Your spectral library as a dataframe.
        new_interval (float): The new interval for resampling.
        method: Interpolation method ('linear', 'cubic', 'quintic', 'akima').
        
    Returns:
        new_df (pd.DataFrame): DataFrame with resampled wavelengths and reflectance values.
    """
    # Extract wavelengths from DataFrame columns
    wavelengths = df.columns.astype(float)
    
    # Create new wavelength range
    new_wavelengths = np.arange(wavelengths[0], wavelengths[-1] + new_interval, new_interval)
    
    # Prepare a new DataFrame for resampled data
    new_df = pd.DataFrame(columns=new_wavelengths)
    
    # Define the interpolation function
    def get_interp_func(wavelengths, reflectance, method):
        if method in ['linear', 'cubic', 'quintic']:
            return interp1d(wavelengths, reflectance, kind=method, fill_value="extrapolate")
        elif method == 'akima':
            from scipy.interpolate import Akima1DInterpolator
            return Akima1DInterpolator(wavelengths, reflectance)
        else:
            raise ValueError("Invalid interpolation method. Choose from 'linear', 'cubic', 'quintic', 'akima'.")
    
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        reflectance = row.values
        interp_func = get_interp_func(wavelengths, reflectance, method)
        new_reflectance = interp_func(new_wavelengths)
        new_df.loc[idx] = new_reflectance
    
    return new_df

def specific_subset(df, wavelength_columns):
    """
    Subsets the dataframe to only include the specified wavelength columns. 
    This was used to visualize the contrast as a result of the PCA analysis. 
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing spectral measurements.
    wavelength_columns (list): A list of wavelength columns to retain in the subset.

    Returns:
    pd.DataFrame: A dataframe containing only the specified wavelength columns.
    """
    # Ensure that the wavelength columns exist in the dataframe
    columns_to_keep = [col for col in wavelength_columns if col in df.columns]
    
    # Subset the dataframe
    subset_df = df[columns_to_keep]
    
    return subset_df

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

#############################################################################
"""PCA ANALYSIS"""

def assign_labels(df, *endmembers):
    """
    Assign labels to each row in the dataframe based on endmember dataframes.
    
    Parameters:
    - df (pd.DataFrame): The full dataset.
    - *endmembers: Variable number of tuples containing endmember dataframe and its label (e.g., (endmember_df, 'Label'))
    
    Returns:
    - labels (np.ndarray): Array of labels for each row in the dataframe.
    - label_mapping (dict): Dictionary mapping label names to numeric values.
    """
    # Initialize labels array with 'mixed'
    labels = np.array(['mixed'] * df.shape[0])
    
    # Assign labels based on the endmember dataframes
    for endmember_df, label in endmembers:
        labels[endmember_df.index] = label
    
    # Convert labels to numeric values
    unique_labels = np.unique(labels)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_mapping[label] for label in labels])
    
    return numeric_labels, label_mapping


def perform_pca_svd(df, n_components=None):
    """
    Perform PCA via SVD and return the variance, 
    the principal component scores, and the loadings. This also plots the singular values
    and cumulative singular values on a log scale.

    Parameters:
    - df (pd.DataFrame): The input dataframe with spectral measurements.
    - n_components (int): Number of principal components to compute. If None, all components are computed.

    Returns:
    - explained_variance (np.ndarray): The variance explained by each principal component.
    - principal_components (pd.DataFrame): The transformed data with principal component scores.
    - loadings (pd.DataFrame): The loadings of the original variables (wavelengths) on the principal components.
    """
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(df_scaled, full_matrices=False)
    
    # Determine the number of components
    if n_components is None:
        n_components = min(df.shape[0], df.shape[1])
    
    # Select the top n_components
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]
    
    # Compute the explained variance
    explained_variance = (S**2) / (df.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    
    # Compute the principal component scores
    principal_components = np.dot(U, np.diag(S))
    principal_components_df = pd.DataFrame(principal_components, 
                                           columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    
    # Compute the loadings
    loadings = Vt.T
    loadings_df = pd.DataFrame(loadings, 
                               columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
                               index=df.columns)
    
    # Plot singular values and cumulative singular values
    plt.figure(figsize=(12, 6))
    
    # Plot singular values
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(S) + 1), S, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Component Number')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Values')
    
    # Plot cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Component Number')
    plt.ylabel('Cumulative Explained Variance (log scale)')
    plt.title('Cumulative Explained Variance')
    
    plt.tight_layout()
    plt.show()
    
    return explained_variance_ratio, principal_components_df, loadings_df


def plot_pca_scatter(principal_components_df, labels, label_mapping, pc1=1, pc2=2, title='PCA Scatter Plot'):
    """
    Plot PCA scatter plot.
    
    Parameters:
    - principal_components_df (pd.DataFrame): DataFrame containing the principal component scores.
    - labels (np.ndarray): Array of numeric labels for each row.
    - label_mapping (dict): Dictionary mapping label names to numeric values.
    - pc1 (int): The principal component to use for the x-axis.
    - pc2 (int): The principal component to use for the y-axis.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(principal_components_df[f'PC{pc1}'], principal_components_df[f'PC{pc2}'], c=labels, cmap='viridis')
    plt.xlabel(f'PC{pc1}')
    plt.ylabel(f'PC{pc2}')
    plt.title(title)
    
    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor='k')
               for label in label_mapping.keys()]
    plt.legend(handles, label_mapping.keys(), loc='best')
    
    plt.show()


def plot_loadings(loadings_df, pc1=1, pc2=2, n_top=10):
    """
    Plot the top n loadings for the specified principal components.
    
    Parameters:
    - loadings_df (pd.DataFrame): DataFrame containing the loadings.
    - pc1 (int): The first principal component to plot.
    - pc2 (int): The second principal component to plot.
    - n_top (int): Number of top loadings to plot for each component.
    """
    pc1_loadings = loadings_df[f'PC{pc1}'].abs().sort_values(ascending=False).head(n_top)
    pc2_loadings = loadings_df[f'PC{pc2}'].abs().sort_values(ascending=False).head(n_top)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    pc1_loadings.plot(kind='bar', color='blue')
    plt.title(f'Top Loadings for PC{pc1}')
    plt.ylabel('Loading Value')
    plt.xlabel('Wavelength')
    
    plt.subplot(1, 2, 2)
    pc2_loadings.plot(kind='bar', color='green')
    plt.title(f'Top Loadings for PC{pc2}')
    plt.ylabel('Loading Value')
    plt.xlabel('Wavelength')
    
    plt.tight_layout()
    plt.show()

def plot_pca_scatter_3d(principal_components_df, labels, label_mapping, pc1=1, pc2=2, pc3=3, title='PCA 3D Scatter Plot'):
    """
    Plot a 3D PCA.
    
    Parameters:
    - principal_components_df (pd.DataFrame): DataFrame containing the principal component scores.
    - labels (np.ndarray): Array of numeric labels for each row.
    - label_mapping (dict): Dictionary mapping label names to numeric values.
    - pc1 (int): The principal component to use for the x-axis.
    - pc2 (int): The principal component to use for the y-axis.
    - pc3 (int): The principal component to use for the z-axis.
    - title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=0., azim=0)
    
    sc = ax.scatter(principal_components_df[f'PC{pc1}'], principal_components_df[f'PC{pc2}'], principal_components_df[f'PC{pc3}'], c=labels, cmap='viridis')
    
    ax.set_xlabel(f'PC{pc1}')
    ax.set_ylabel(f'PC{pc2}')
    ax.set_zlabel(f'PC{pc3}')
    ax.set_title(title)
    
    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor='k')
               for label in label_mapping.keys()]
    ax.legend(handles, label_mapping.keys(), loc='best')
    
    plt.show()


#############################################################################
"""QHULL"""

#Don't run this without reducing dimensions... LOL
def compute_convex_hull(df):
    """
    This function computes the convex hull for the given pandas DataFrame.
    It identifies which entries in the DataFrame were chosen as vertices of the convex hull.
    
    Parameters:
    df (pandas.DataFrame): Your spectral library as a dataframe
    
    Returns:
    hull_vertices (list): Indices of the DataFrame entries chosen as vertices of the convex hull.
    """
    
    # Convert DataFrame to NumPy array
    points = df.to_numpy()
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Return the indices of the vertices
    hull_vertices = hull.vertices
    return hull_vertices


"""TRIANGLES"""

#The shoelace formula is derived from the fact that the area of a triangle can be represented as half of the absolute value of the determinant of the coordinates of its vertices
def area_of_triangle(p1, p2, p3):
    """Calculate the area of a triangle given its vertices using the Shoelace formula."""
    return 0.5 * abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[1]*p2[0] - p2[1]*p3[0] - p3[1]*p1[0])

def compute_three_point_hull_max(df):
    """
    This function selects exactly three points from the given pandas DataFrame
    that form the largest possible triangle.
    
    Parameters:
    df (pandas.DataFrame): Your spectral library as a dataframe.
    
    Returns:
    selected_indices (list): Indices of the DataFrame entries chosen as the three vertices.
    """
    # Convert DataFrame to NumPy array
    points = df.to_numpy()
    
    # Ensure we have at least three points
    if len(points) < 3:
        raise ValueError("DataFrame must contain at least three points")
    
    max_area = 0
    selected_indices = None
    
    # Iterate over all combinations of three points to find the maximum enclosing triangle
    for comb in combinations(range(len(points)), 3):
        p1, p2, p3 = points[comb[0]], points[comb[1]], points[comb[2]]
        area = area_of_triangle(p1, p2, p3)
        if area > max_area:
            max_area = area
            selected_indices = comb
    
    return list(selected_indices)


def compute_three_point_hull_min(df):
    """
    This function selects exactly three points from the given pandas DataFrame
    that form the minimal enclosing triangle in the original high-dimensional space.
    
    Parameters:
    df (pandas.DataFrame): Your spectral library as a dataframe.
    
    Returns:
    selected_indices (list): Indices of the DataFrame entries chosen as the three vertices.
    """
    # Convert DataFrame to NumPy array
    points = df.to_numpy()
    
    min_area = float('inf')
    selected_indices = None
    
    # Iterate over all combinations of three points to find the minimal enclosing triangle
    for comb in combinations(range(len(points)), 3):
        p1, p2, p3 = points[comb[0]], points[comb[1]], points[comb[2]]
        # Calculate the area of the triangle formed by the three points
        area = area_of_triangle(p1, p2, p3)
        if area < min_area:
            min_area = area
            selected_indices = comb
    
    return list(selected_indices)

# =============================================================================
# Linear Mixing Model 
# =============================================================================
#fractional abundance estimator
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
    '''
    The objective function provides the optimizer (result = minimize) with a measure of how far the guessed spectra is from the inputted spectra.
    It does this by minimizing the MSE between the two. 
    
    First the fractions array is reshaped so that its dimensions are combatible with the number of endmembers defined.
    The index of -1 tells numpy to automatically reshape the array. Which in this case is the number of spectra multiplied by the number of endmembers.
    Therefore you can optimize the selection of endmembers based on many sample mixed spectrums and have the dot product operation be valid.
    The dot product is what actually computes the estimates (reconstructed) spectrum to fit the true spectrum. 
    This comes from the assumption that the mixture is linear and thus a weighted sum of the sprectra in an endmember by their abundances.
    
    The MSE is then calculated to feed back into the optimization.
    
    '''

    def constraint_sum_to_one(fractions):
        # Constraint: sum of fractions should be 1
        return np.sum(fractions) - 1
    
    def constraint_greater_than_zero(fractions):
        # Constraint: fractions should be greater than zero
        return fractions - epsilon
    
    '''
    The constraints are necessary for the optimization process so that the optimized abundances make physical sense.
    The contribution of an endmember is  a spectrum cannot be negative.
    Furthermore, all endmembers in a scene should contribute to the reconstructed spectra and the total abundances must equal 100%.
    
    The sum to one constraint function works by checking to see if the residual between the sum of the fractional abundaces and 1 is equal to 0.
    The result is passed on to the minimize operator which has an 'eq' clause in the constraints parameter. This means only solutions where the sum_to_one function passes a 0 are valid solutions.
    
    The greater_than_zero function works by subtracting epsilon, which is an arbitrarily small number, from the estimated fraction.
    If the value is negative it, the corresponding solution will be rejected in the minimize operator via the "ineq" clause in the constrainst parameter.
    '''
    
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
        
        '''
        The initial guess just makes the contributions of all endmembers equal for each band in the spectrum.
        This is a good starting point for the minimize operator since it satisfies all constraints and can be any size depending on the number if endmember and spectra to fit for.
        '''

        # Constraints for the current spectrum
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                       {'type': 'ineq', 'fun': constraint_greater_than_zero}]
        
        '''
        How to enforce the constraints are defined above and are given as parameters in the minimize operator.
        '''

        # Optimization for the current spectrum
        result = minimize(objective, initial_guess, args=(spectrum, endmembers_array),
                          constraints=constraints, method='SLSQP', options={'disp': False})
        
        '''
        This minimize operator is the core of the function and uses the scipy.optimize library.
        In this function we want to find the fractional abundances that minimize the RMSE between the observed and reconstructed spectra using n endmembers.
        
        The objective is the function we want to minimize which is described above.
        
        The initial guess gives us the starting point for optimization described above
        
        The args are passed to the objective function and come from the spectra and endmembers pd.Dataframes which are supplied in the LMM_FAE parameters.
       
        The constraints are given.
        
        The Sequential Least Squares Programming (method = SLSQP) iterively checks combinations of fractional abundances and chooses new abundances.
        This is basically a gradient descent which allows for constraints to be placed so that non physically possible variables are not chosen (such as negative fractional abundances).
        This way, we are not randomly changing the fill fractions but instead are calculating the gradient across nearby combinations of fractions.
        The "topography" of the RMSE space is shown through the gradient and the fractions which correspond to the largest change in slope minimizing the RMSE are chosen for the next solution option.
        The gradient itself is calculated numerically through the finite difference method.
        The numeric solution is calculated whilst considering the constraints.
        
        
        options={'disp': False} does not print each step of the minimization process.
        '''

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
            plt.show()
    
    # Convert the final estimated fractions back to a DataFrame
    fractions_df = pd.DataFrame(all_fractions, index=spectra.index, columns=endmembers.index)
    
    # Create a Series for individual RMSE values
    individual_rmse_series = pd.Series(individual_rmse, index=spectra.index, name='RMSE')

    # Calculate the overall RMSE across all spectra
    overall_rmse = np.mean(individual_rmse)
    
    return fractions_df, overall_rmse, individual_rmse_series



#Endmember optimizer
def LMM_OE(R_obs, df_crop_residue, df_vegetation, df_soil):
    """
    Optimize the fractional abundances of three endmembers (Crop residue, Vegetation, Soil) 
    that minimize the RMSE for a single mixed spectra.

    Parameters:
    R_obs (np.ndarray): 1D array of observed reflectance values.
    df_crop_residue (pd.DataFrame): DataFrame containing crop residue endmembers.
    df_vegetation (pd.DataFrame): DataFrame containing vegetation endmembers.
    df_soil (pd.DataFrame): DataFrame containing soil endmembers.

    Returns:
    dict: A dictionary containing the following:
        - 'abundances': Optimized fractional abundances for the chosen endmembers.
        - 'chosen_endmembers': The chosen endmembers (row indices) from each DataFrame.
        - 'rmse': The RMSE of the optimal combination.
    
    This function optimizes fractional abundances for three endmembers (crop residue, vegetation, soil) based on single observed spectra. 
    It compares the RMSE for every possible combination of endmembers.
    It loops through all combinations of endmembers and solves a constrained optimization problem for each combination.
    Therefore you can have many different endmember spectrum for each endmember type. 
    The function will pick the best combination.
    Constraints (sum-to-one and non-negativity) are applied directly in each optimization call.
    The optimizer handles a small number of variables (3 fractions for 3 endmembers).
    """
    best_rmse = float('inf')
    best_combination = None
    best_fractions = None
    
    # Iterate over all combinations of one endmember from each DataFrame
    for crop_idx in df_crop_residue.index:
        for veg_idx in df_vegetation.index:
            for soil_idx in df_soil.index:
                # Extract the current endmembers as numpy arrays
                E_crop = df_crop_residue.loc[crop_idx].values
                E_veg = df_vegetation.loc[veg_idx].values
                E_soil = df_soil.loc[soil_idx].values
                
                # Define the objective function to minimize RMSE
                def rmse(fractions):
                    reconstructed_spectra = (fractions[0] * E_crop + 
                                         fractions[1] * E_veg + 
                                         fractions[2] * E_soil)
                    return np.sqrt(np.mean((R_obs - reconstructed_spectra) ** 2))
                
                # Constraints: Fractions should sum to 1
                constraints = {'type': 'eq', 'fun': lambda f: np.sum(f) - 1}
                
                # Bounds: Each fraction should be between 0 and 1
                bounds = [(0, 1), (0, 1), (0, 1)]
                
                # Initial guess: Equal fractions
                initial_guess = np.array([1/3, 1/3, 1/3])
                
                # Perform optimization
                result = minimize(rmse, initial_guess, constraints=constraints, bounds=bounds)
                
                # Check if this combination is the best so far
                if result.fun < best_rmse:
                    best_rmse = result.fun
                    best_combination = (crop_idx, veg_idx, soil_idx)
                    best_fractions = result.x

    return {
        'abundances': best_fractions,
        'chosen_endmembers': best_combination,
        'rmse': best_rmse
    }

# =============================================================================
"""Loading and Defining Data"""
# =============================================================================

#load spectral library csv using pandas. You can use os methods to make this cleaner I guess.
lib = pd.read_csv("C:\\Users\\Aidan\\OneDrive - University of Toronto\\Desktop\\UTAT\\Science\\ENDMEMBER\\Data\\UTAH\\fractional-cover-simulated-vswir-dataset-version-2--original-10nm-spectra.csv")

lib_fa = lib.iloc[:, [2, 3, 5]]

#setting up with vegetation test
veg_no_em = lib[(lib.iloc[:, 3] <= .90) & (lib.iloc[:, 5] <= .90) & (lib.iloc[:, 2] <= .90)] #subset needed
veg_no_em_fa = veg_no_em.iloc[:, [2, 3, 5]]

#extract endmembers with vegetation
vegetation = endmember(lib, "gv_fraction")
npv = endmember(lib, "npv_fraction") #note that NPV refers to crop residue and stands for Non-Photosynthetic Vegetation.... LOL
soil = endmember(lib, "soil_fraction")


#setting up without vegetation test
no_veg_lib = lib[lib.iloc[:, 2] == 0] #subset need
no_veg_lib_fa = no_veg_lib.iloc[:, [3, 5]]
no_veg_no_em = no_veg_lib[(no_veg_lib.iloc[:, 3] <= .90) & (no_veg_lib.iloc[:, 5] <= .90)] #subset needed
no_veg_no_em_fa = no_veg_no_em.iloc[:, [3, 5]]

#extract endmemebrs without vegetation
no_veg_npv = endmember(no_veg_lib, "npv_fraction")
no_veg_soil = endmember(no_veg_lib, "soil_fraction")



# =============================================================================
"""Subsetting"""
# =============================================================================

"""WITH VEGETATION"""
#Subsetting with veg test with and without endmembers 
reflectance_lib = refvalues(lib)
reflectance_lib_no_em = refvalues(veg_no_em)
veg_reflection = refvalues(vegetation)
npv_reflection = refvalues(npv)
soil_reflection = refvalues(soil)


#subset to FINCH spectral range
FINCH_lib_ref = broad_subset(reflectance_lib, 50, 130)
FINCH_lib_ref_no_em = broad_subset(reflectance_lib_no_em, 50, 130)
FINCH_veg_ref = broad_subset(veg_reflection, 50, 130)
FINCH_npv_ref = broad_subset(npv_reflection, 50, 130)
FINCH_soil_ref = broad_subset(soil_reflection, 50, 130)

#subset to Raymond Feature
raymond_lib_ref = broad_subset(reflectance_lib, 150, 250)
raymond_lib_ref_no_em = broad_subset(reflectance_lib_no_em, 150, 250)
raymond_veg_ref = broad_subset(veg_reflection, 150, 250)
raymond_npv_ref = broad_subset(npv_reflection, 150, 250)
raymond_soil_ref = broad_subset(soil_reflection, 150, 250)

"""NO VEGETATION"""
#Subsetting without veg test with and without endmembers
reflectance_no_veg_lib = refvalues(no_veg_lib)
reflectance_no_veg_no_em_lib = refvalues(no_veg_no_em)
no_veg_npv_reflection = refvalues(no_veg_npv)
no_veg_soil_reflection = refvalues(no_veg_soil)

#subset to FINCH spectral range
no_veg_FINCH_lib_ref = broad_subset(reflectance_no_veg_lib, 50, 130)
no_veg_FINCH_lib_ref_no_em = broad_subset(reflectance_no_veg_no_em_lib, 50, 130)
no_veg_FINCH_npv_ref = broad_subset(no_veg_npv_reflection, 50, 130)
no_veg_FINCH_soil_ref = broad_subset(no_veg_soil_reflection, 50, 130)

#subset to Raymond Feature
no_veg_raymond_lib_ref = broad_subset(reflectance_no_veg_lib, 150, 250)
no_veg_raymond_lib_ref_no_em = broad_subset(reflectance_no_veg_no_em_lib, 150, 250)
no_veg_raymond_npv_ref = broad_subset(no_veg_npv_reflection, 150, 250)
no_veg_raymond_soil_ref = broad_subset(no_veg_soil_reflection, 150, 250)

# =============================================================================
"""Resampling"""
# =============================================================================

"""WITH VEGETATION"""
lib_resampled = resample_reflectance_df(reflectance_lib, 10, method='cubic')
resampled_reflectance_lib_no_em = resample_reflectance_df(reflectance_lib_no_em, 10, method='cubic')
resampled_veg_reflection = resample_reflectance_df(veg_reflection, 10, method='cubic')
resampled_npv_reflection = resample_reflectance_df(npv_reflection, 10, method='cubic')
resampled_soil_reflection = resample_reflectance_df(soil_reflection, 10, method='cubic')

FINCH_resampled = resample_reflectance_df(FINCH_lib_ref, 10, method='cubic')
no_em_FINCH_resampled = resample_reflectance_df(FINCH_lib_ref_no_em, 10, method='cubic')
FINCH_veg_resampled = resample_reflectance_df(FINCH_veg_ref, 10, method='cubic')
FINCH_soil_resampled = resample_reflectance_df(FINCH_soil_ref, 10, method='cubic')
FINCH_npv_resampled = resample_reflectance_df(FINCH_npv_ref, 10, method='cubic')

raymond_resampled = resample_reflectance_df(raymond_lib_ref, 10, method='cubic')
no_em_raymond_resampled = resample_reflectance_df(raymond_lib_ref_no_em, 10, method='cubic')
raymond_veg_ref_resampled = resample_reflectance_df(raymond_veg_ref, 10, method='cubic')
raymond_npv_ref_resampled = resample_reflectance_df(raymond_npv_ref, 10, method='cubic')
raymond_soil_ref_resampled = resample_reflectance_df(raymond_soil_ref, 10, method='cubic')



"""NO VEGETATION"""
no_veg_lib_resampled = resample_reflectance_df(reflectance_no_veg_lib, 10, method='cubic') 
no_veg_no_em_lib_resampled = resample_reflectance_df(reflectance_no_veg_no_em_lib, 10, method='cubic')
no_veg_npv_resampled = resample_reflectance_df(no_veg_npv_reflection, 10, method='cubic') 
no_veg_soil_resampled = resample_reflectance_df(no_veg_soil_reflection, 10, method='cubic') 

#subset to FINCH spectral range
no_veg_FINCH_lib_resampled = resample_reflectance_df(no_veg_FINCH_lib_ref, 10, method='cubic') 
no_veg_no_em_FINCH_lib_resampled = resample_reflectance_df(no_veg_FINCH_lib_ref_no_em, 10, method='cubic') 
no_veg_FINCH_npv_resampled = resample_reflectance_df(no_veg_FINCH_npv_ref, 10, method='cubic') 
no_veg_FINCH_soil_resampled = resample_reflectance_df(no_veg_FINCH_soil_ref, 10, method='cubic') 

#subset to Raymond Feature
no_veg_raymond_lib_resampled = resample_reflectance_df(no_veg_raymond_lib_ref, 10, method='cubic') 
no_veg_no_em_raymond_lib_resampled = resample_reflectance_df(no_veg_raymond_lib_ref_no_em, 10, method='cubic')
no_veg_raymond_npv_resampled = resample_reflectance_df(no_veg_raymond_npv_ref, 10, method='cubic') 
no_veg_raymond_soil_resampled = resample_reflectance_df(no_veg_raymond_soil_ref, 10, method='cubic') 

# =============================================================================
"""Inversion"""
# =============================================================================
# 1447, 73, 215              17

# Inversion #1: FINCH Library with vegetation and with endmebers

