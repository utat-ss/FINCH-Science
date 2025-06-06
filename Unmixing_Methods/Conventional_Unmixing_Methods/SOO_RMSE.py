import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable

def SOO_RMSE(spectra: pd.DataFrame, endmembers: pd.DataFrame, plot_results: bool = False):
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

def SOO_RMSE_Randomized(spectra: pd.DataFrame, 
            endmembers: pd.DataFrame, 
            plot_results: bool = False, 
            n_starts: int = 5,
            initial_guess_fn: Callable[[int], np.ndarray] = None):
    """
    Estimate fractional abundances with sum-to-one and greater-than-zero constraints 
    using a linear spectral mixing model. Uses multi-start SLSQP optimization to avoid local minima.

    Parameters:
    - spectra (pd.DataFrame): Rows = mixed spectra, columns = wavelengths.
    - endmembers (pd.DataFrame): Rows = endmember spectra, columns = wavelengths.
    - plot_results (bool): If True, plots original vs reconstructed spectra.
    - n_starts (int): Number of random initial guesses per spectrum.

    Returns:
    - fractions_df: DataFrame of estimated fractional abundances.
    - overall_rmse: Mean RMSE across all spectra.
    - individual_rmse_series: Per-spectrum RMSEs.
    """
    def objective(fractions, spectrum, endmembers_array):
        reconstructed = np.dot(fractions, endmembers_array)
        mse = np.mean((spectrum - reconstructed) ** 2)
        return mse

    def constraint_sum_to_one(fractions):
        return np.sum(fractions) - 1

    def constraint_greater_than_zero(fractions):
        return fractions - epsilon

    # Setup
    spectra_array = spectra.values
    endmembers_array = endmembers.values
    num_spectra = spectra_array.shape[0]
    num_endmembers = endmembers_array.shape[0]
    epsilon = 1e-6

    all_fractions = np.zeros((num_spectra, num_endmembers))
    individual_rmse = []

    for i in range(num_spectra):
        spectrum = spectra_array[i, :]
        best_mse = np.inf
        best_fractions = None

        for _ in range(n_starts):
            # Use custom initial guess function, or default to Dirichlet
            if initial_guess_fn is not None:
                initial_guess = initial_guess_fn(num_endmembers)
            else:
                initial_guess = np.random.dirichlet(np.ones(num_endmembers))

            constraints = [
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'ineq', 'fun': constraint_greater_than_zero}
            ]

            best_local = {'mse': np.inf, 'fractions': None}
            def callback(xk):
                mse = objective(xk, spectrum, endmembers_array)
                if mse < best_local['mse']:
                    best_local['mse'] = mse
                    best_local['fractions'] = xk.copy()

            result = minimize(
                objective,
                initial_guess,
                args=(spectrum, endmembers_array),
                constraints=constraints,
                method='SLSQP',
                callback=callback,
                options={'disp': False, 'ftol': 1e-12, 'maxiter': 1000}
            )

            # Use callback result if available, otherwise fallback
            current_fractions = best_local['fractions'] if best_local['fractions'] is not None else result.x
            current_mse = objective(current_fractions, spectrum, endmembers_array)

            if current_mse < best_mse:
                best_mse = current_mse
                best_fractions = current_fractions

        # Store best result for this spectrum
        all_fractions[i, :] = best_fractions
        reconstructed = np.dot(best_fractions, endmembers_array)
        rmse = np.sqrt(np.mean((spectrum - reconstructed) ** 2))
        individual_rmse.append(rmse)

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(spectra.columns, spectrum, label='True Spectrum', color='blue')
            plt.plot(spectra.columns, reconstructed, label='Reconstructed Spectrum', color='red', linestyle='--')
            plt.title(f'Spectrum {i+1}: True vs Reconstructed')
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.legend()
            plt.show()

    # Final results
    fractions_df = pd.DataFrame(all_fractions, index=spectra.index, columns=endmembers.index)
    individual_rmse_series = pd.Series(individual_rmse, index=spectra.index, name='RMSE')
    overall_rmse = np.mean(individual_rmse)

    return fractions_df, overall_rmse, individual_rmse_series
