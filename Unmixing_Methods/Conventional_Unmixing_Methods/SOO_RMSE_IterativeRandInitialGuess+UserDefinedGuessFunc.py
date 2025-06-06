import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable


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

#example random guess function and use:
def random_uniform_simplex(n):
    x = np.random.rand(n)
    return x / np.sum(x)

# Usage
# fractions_df, rmse_avg, rmse_each = LMM_FAE(
#     spectra,
#     endmembers,
#     n_starts=10,
#     initial_guess_fn=random_uniform_simplex
# )