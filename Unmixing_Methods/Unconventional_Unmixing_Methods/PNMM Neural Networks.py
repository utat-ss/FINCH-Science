# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 00:35:41 2025

The purpose of this code is to create a Neural Network to predict how the spectral signature of crop residue
(the non-photosynthetic vegetation) changes over time as it decomposes into soil.

Theory of Transformation
1. Physical Analysis:
As crop residue (non-photosynthetic vegetation, NPV) decompose, their spectral properties change over time
into soil. This transformation is governed by physical and chemical processes such as organic matter decay,
moisture loss, and mineral exposure.

To represent this transformation take the observed spectrum as a total linear or non-linear mixture of spectral
signatures (NPV endmember and soil endmember). The transformation factor/fraction (Decay function of time) α(t)
represents the degree of decay/rate of decay at time t.
    
Imagine you have a smoothie made of two fruits. One being an unripe bitter banana representing
non-photosynthetic vegetation (NPV) and the other being soil a super over ripe sweet and mushy banana.
At the beginning, the smoothie is mostly NPV but over time left out in the environment the smoothie starts
ripening and now more of the smoothie is soil. Now the smoothie has a different taste and colour.

In a linear model, a mixture with NPV and soil would simply be those two as strict two amounts. Like 20% soil and 80% NPV.
In a non-linear model it is more complex. Now we observe and learn how the spectral signature of NPV changes into soil
(how the colour of a material's wavelength changes). 
Linear is a predictable way of the soil and npv changing at a constant rate and non-linear is not constant rate. In real
world scenerio its almost always non-linear
 
So FINCH sees the ground as a spectrum, the *smoothie*. This mixed smoothie (spectrum) is a blend of NPV
and soil signal.

Recipe:

NPV Signal: This is the initial ingredient
Soil Signal: This is like the ingredient that increases slowly with time (banana ripening)
Mixing Ratio α(t): This function tells us how much soil signal is in the mix at any time t.
Meaning if t is small (early stages), α(t) is low because there's little soil early on. With
time going pass, α(t) increases as more soil appears in the mix, more bananas ripening in the
smoothie. Can be written as: 

                Spectrum(λ, t) = α(t) * Soil(λ) + (1 - α(t)) * NPV(λ) + ε(t) where 0<=α(t)<=1

Where α(t) increases with time t.

NPV: Spectral Signature
Soil: Spectral Signature
α(t): Decay function of time. If t = 0 no decay has happened, meaning α(0) = thus the observed spectrum is 100%
      As time increases α(t) increases toward 1. If α(t) = 1 then the observed spectrum is 100% soil.
      Can change in a linear or non-linear way depending on the decay processes.
α(t) * Soil: This tells you how much of the soil's spectral signature is present int he overall observed spectrum,
             α(t) > 0 The more soil
(1 - α(t)) * NPV: The remaining fraction. So it tells you how much of the NPV's spectral signature is still there. 1 is
            a ratio to 100%
Spectrum(t): FINAL OBSERVED SPECTRUM!! At time t :)
             It is the combination of the two defined proportions α(t) * Soil and (1 - α(t)) * NPV, respectively


2. Incorporating Physics into Data Augmentation:
Spectral Preprocessing is for when we receive raw hyperspectral data. Raw data is noisy as there are many unwanted
materials FINCH collects. it contains too much information! The important details such as the decay process
(transformation of NPV to soil over time) is hard to see.

To fix this we use techniques like
Continuum Removal: A process where it helps clear/eliminate noise in the spectrum so subtle features become more vivid.
Normalization: This scales the data in a way so that differences between the overall brightness do not mask or cover the
               important details we need to know.
Derivative Analysis: Creates changes in the spectral curves, revealing small changes in the material's properties as decay
                     progresses.
Principal Componenet Analysis: 
They give us a cleaner data!

@author: Tomi Wang :)
"""

import pandas as pd # This is a library for data analysis
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from math import factorial
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error

# Data Loading
# ---------------------------------------
def data_loading(file_path = None):
    """
    The code loading our csv hyperspectral data files
    """
    if file_path is None: # Checks if the file path parameter was provided when the function is called
        file_path = input("Enter file path here: ") # If no file it will prompt to add file
    try:
        df = pd.read_csv(file_path) # Try to read the csv file from a dataframe
        print("Data loaded 👍")
        return df
    except Exception as e: # Error handling
        print("Error loading data file: ", e)
        return None
    
def get_abundance_columns(df, abundance_cols=None):
    if abundance_cols is None:
        abundance_input = input("Enter the abundance, separated by commas: ")
        abundance_cols = [col.strip() for col in abundance_input.split(",")]
    try:
        selected_data = df[abundance_cols]
        print("Abundance Data Selected 👍")
        return selected_data
    except KeyError as e:
        print("Error: One or more abundance columns not found:", e)
        return None

def get_wavelength_columns(df, wavelength_input=None):
    if wavelength_input is None:
        wavelength_input = input("Enter wavelength values or range: ")
    available_wavelengths = [col for col in df.columns if col.isdigit()]
    selected_wavelengths = set()
    parts = wavelength_input.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                selected_wavelengths.update([col for col in available_wavelengths if start <= int(col) <= end])
            except ValueError:
                print(f"Invalid range input: {part}. Skipping.")
        else:
            try:
                val = int(part)
                if str(val) in available_wavelengths:
                    selected_wavelengths.add(str(val))
                else:
                    print(f"Wavelength {val} not found in data. Skipping.")
            except ValueError:
                print(f"Invalid input: {part}. Skipping.")
    if not selected_wavelengths:
        print("No valid wavelengths selected")
        return None
    print("Wavelength Data Selected 👍")
    return df[list(selected_wavelengths)]


# Preprocesssing
# ---------------------------------------
def prepare_matrices(soil, npv, alphas, epsilon=0):
    A = np.vstack([soil, npv]).T
    Y = np.array([alpha * soil + (1 - alpha) * npv + epsilon for alpha in alphas])
    return A, Y

def continuum_removal(spectrum):
    wavelengths = np.arange(len(spectrum))
    continuum = np.interp(wavelengths, [0, len(spectrum)-1], [spectrum[0], spectrum[-1]])
    return spectrum / continuum

def normalize_spectrum(spectrum):
    return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

def derivative_analysis(spectrum):
    return np.gradient(spectrum)

def principle_componenet_analysis(spectra, n_components=5):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(spectra)


# Endmember Extraction
# ---------------------------------------
def simplex_volume(endmembers):
    m = endmembers.shape[0] - 1
    V = endmembers[1:] - endmembers[0]
    G = np.dot(V, V.T)
    return np.sqrt(np.abs(np.linalg.det(G))) / factorial(m)

def n_findr(data, num_endmembers, max_iter=100, tol=1e-6):
    n = data.shape[0]
    idx = np.random.choice(n, num_endmembers, replace=False)
    endmembers = data[idx].copy()
    current_vol = simplex_volume(endmembers)
    for _ in range(max_iter):
        improved = False
        for i in range(num_endmembers):
            for j in range(n):
                candidate = endmembers.copy()
                candidate[i] = data[j]
                vol_candidate = simplex_volume(candidate)
                if vol_candidate > current_vol + tol:
                    endmembers[i] = data[j]
                    current_vol = vol_candidate
                    improved = True
        if not improved:
            break
    return endmembers


# Linear Mixing Model
# ---------------------------------------
def fully_constrained_least_squares(y, E):
    n = E.shape[1]
    def objective(a):
        return np.linalg.norm(y - E.dot(a))**2
    constraints = [{'type': 'eq', 'fun': lambda a: np.sum(a) - 1},
                   {'type': 'ineq', 'fun': lambda a: a}]
    a0 = np.ones(n) / n
    result = minimize(objective, a0, constraints=constraints, method='SLSQP')
    return result.x

def linear_mixing_model(soil, npv, alpha, epsilon=0):
    alpha = np.clip(alpha, 0, 1)
    return alpha * soil + (1 - alpha) * npv + epsilon


# Neural Network
# ---------------------------------------
def preprocess_spectral_input(spectrum):
    print("Preprocessing Spectral Inputs Complete 👍")
    return tf.convert_to_tensor(np.array(spectrum, dtype=np.float32))

def hidden_layers(input_layer, hidden_units=[64, 32]):
    x = input_layer
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
    print("Hidden Layers Complete 👍")
    return x

def output_layer(hidden_layer):
    print("Output Layers Complete 👍")
    return layers.Dense(1, activation='linear')(hidden_layer)

def compile_model(model):
    optimizer_instance = get_optimizer()
    model.compile(optimizer=optimizer_instance, loss='mse', metrics=['mae'])
    print("Compiling Complete 👍")
    return model

def get_optimizer():
    print("Optimizer Complete 👍")
    return Adam(learning_rate=0.001)

def create_model(input_shape):
    input_layer = keras.Input(shape=(input_shape,))
    hidden_layer = hidden_layers(input_layer)
    model_output = output_layer(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=model_output)
    print("Creating Model Complete 👍")
    return compile_model(model)


# Training Model
# ---------------------------------------
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

def validate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=1)

def k_fold_cross_validation(X, y, k=5, epochs=50, batch_size=32):
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    if isinstance(y, tf.Tensor):
        y = y.numpy()
        
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    losses = []
    maes = []
    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"Training fold {fold}...")
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        model = create_model(input_shape=X.shape[1])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, verbose=1)
        results = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} - Loss: {results[0]:.4f}, MAE: {results[1]:.4f}")
        losses.append(results[0])
        maes.append(results[1])
        fold += 1
    return losses, maes


# Accuracy
# ---------------------------------------
def accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = r2_score(y_test, predictions) * 100
    print("Neural Network is predicting the non linear mixing model with {:.2f}% accuracy".format(accuracy))
    return accuracy

# make one for linear


# Visualizations
# ---------------------------------------
def plot_spectral_curves(A, Y):
    plt.figure()
    num_wavelengths, num_endmembers = A.shape
    x_indices = np.arange(num_wavelengths)
    for i in range(num_endmembers):
        plt.plot(x_indices, A[:, i], marker='o', label=f'Endmember {i+1}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Reflectance')
    plt.title('Spectral Curves for Matrix A (Endmembers)')
    plt.legend()
    plt.show()

    plt.figure()
    num_mixes, num_wavelengths_y = Y.shape
    x_indices_y = np.arange(num_wavelengths_y)
    for i in range(num_mixes):
        plt.plot(x_indices_y, Y[i, :], marker='o', label=f'Mix {i+1}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Reflectance')
    plt.title('Spectral Curves for Matrix Y (Mixed Spectra)')
    plt.legend()
    plt.show()

def plot_nfindr_2d(pca_data, endmembers):
    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], label='Data Points')
    plt.scatter(endmembers[:, 0], endmembers[:, 1], marker='x', s=100, c='red', label='Endmembers')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('2D Scatter Plot with Endmembers')
    plt.legend()
    plt.show()


def plot_nfindr_3d(pca_data, endmembers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], label='Data Points')
    ax.scatter(endmembers[:, 0], endmembers[:, 1], endmembers[:, 2], marker='x', s=100, c='red', label='Endmembers')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.title('3D Scatter Plot with Endmembers')
    plt.legend()
    plt.show()

def plot_mixing_coefficients(coeffs):
    indices = np.arange(len(coeffs))
    plt.figure()
    plt.bar(indices, coeffs)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Abundance')
    plt.title('Mixing Coefficients Estimated')
    plt.xticks(indices)
    plt.show()

def plot_linear_mixing_model(soil, npv, alpha):
    mixed = alpha * soil + (1 - alpha) * npv
    wavelengths = np.arange(len(soil))
    plt.figure()
    plt.plot(wavelengths, soil, marker='o', label='Soil Spectrum')
    plt.plot(wavelengths, npv, marker='o', label='NPV Spectrum')
    plt.plot(wavelengths, mixed, marker='o', label=f'Mixed Spectrum (α={alpha})')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Reflectance')
    plt.title('Linear Mixing Model Comparison')
    plt.legend()
    plt.show()

def plot_heatmaps(A, Y):
    plt.figure()
    plt.imshow(A, aspect='auto', interpolation='none')
    plt.colorbar(label='Reflectance')
    plt.xlabel('Endmember Index')
    plt.ylabel('Wavelength Index')
    plt.title('Heatmap of Matrix A (Endmembers)')
    plt.show()

    plt.figure()
    plt.imshow(Y, aspect='auto', interpolation='none')
    plt.colorbar(label='Reflectance')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Mix Index (α value)')
    plt.title('Heatmap of Matrix Y (Mixed Spectra)')
    plt.show()

def plot_spectrum_continuum_removal(spectrum):
    wavelengths = np.arange(len(spectrum))
    cr_spectrum = continuum_removal(spectrum)
    plt.figure()
    plt.plot(wavelengths, spectrum, label='Original Spectrum')
    plt.plot(wavelengths, cr_spectrum, label='Continuum Removed')
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.show()

def plot_spectrum_normalization(spectrum):
    norm_spectrum = normalize_spectrum(spectrum)
    wavelengths = np.arange(len(spectrum))
    plt.figure()
    plt.plot(wavelengths, spectrum, label='Original Spectrum')
    plt.plot(wavelengths, norm_spectrum, label='Normalized Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_spectrum_derivative(spectrum):
    derivative = derivative_analysis(spectrum)
    wavelengths = np.arange(len(spectrum))
    plt.figure()
    plt.plot(wavelengths, spectrum, label='Original Spectrum')
    plt.plot(wavelengths, derivative, label='Derivative')
    plt.xlabel('Wavelength')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_pca_2d(pca_data):
    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('2D PCA Scatter Plot')
    plt.show()

def plot_pca_3d(pca_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2])
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.title('3D PCA Scatter Plot')
    plt.show()

def plot_loss(history):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. Actual Values")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

def plot_kfold_results(losses, maes):
    folds = np.arange(1, len(losses) + 1)
    width = 0.35
    plt.figure()
    plt.bar(folds - width / 2, losses, width=width, label='Loss')
    plt.bar(folds + width / 2, maes, width=width, label='MAE')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title('K-Fold Cross Validation Performance')
    plt.xticks(folds)
    plt.legend()
    plt.show()


if __name__ == '__main__':
# Functions Called
# ---------------------------------------
    data = data_loading()
    
    if data is not None:
        # Select abundance and wavelength data
        abundance_data = get_abundance_columns(data)
        wavelength_data = get_wavelength_columns(data)
        
        if abundance_data is not None and wavelength_data is not None:
            print("\nAbundance Data:")
            print(abundance_data.head())
            print("\nWavelength Data:")
            print(wavelength_data.head())

            # Convert DataFrames to numpy arrays
            abundance_data = abundance_data.to_numpy()
            wavelength_data = wavelength_data.to_numpy()

            # Apply Preprocessing Steps
            print("Applying Preprocessing Steps...")
            preprocessed_data = []
            for spectrum in wavelength_data:
                spectrum = continuum_removal(spectrum)
                spectrum = normalize_spectrum(spectrum)
                spectrum = derivative_analysis(spectrum)
                preprocessed_data.append(spectrum)
            
            preprocessed_data = np.array(preprocessed_data)
            print("Preprocessing Completed! 👍")

            # Apply PCA
            print("Applying PCA for dimensionality reduction...")
            reduced_data = principle_componenet_analysis(preprocessed_data, n_components=5)
            print("PCA Completed! 👍")

            # Convert to Tensor
            X = preprocess_spectral_input(reduced_data)
            y = preprocess_spectral_input(abundance_data)

            # Model Creation
            model = create_model(input_shape=X.shape[1])

            # Train Model
            print("Training Model...")
            train_model(model, X, y, epochs=50, batch_size=32)

            # Validate Model
            print("Validating Model...")
            validate_model(model, X, y)

            # K-Fold Cross Validation
            print("Performing K-Fold Cross Validation...")
            losses, maes = k_fold_cross_validation(X, y, k=5, epochs=50, batch_size=16)
            print("Cross Validation Losses:", losses)
            print("Cross Validation MAEs:", maes)
            
            soil = np.array([0.7, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88])
            npv = np.array([0.3, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48])
            alphas = np.linspace(0, 1, num=10)
            A, Y = prepare_matrices(soil, npv, alphas)
            print("Mixing matrix A:")
            print(A)
            print("Observed spectra Y:")
            print(Y)
            
            endmembers = n_findr(Y, 2)
            print("Extracted endmembers:")
            print(endmembers)
            
            y_sample = Y[5]
            a_estimated = fully_constrained_least_squares(y_sample, A)
            print("Estimated abundances (mixing coefficients):")
            print(a_estimated)
            
            alpha_val = 0.6
            spectrum_mix = linear_mixing_model(soil, npv, alpha_val)
            print("Linear mixing model output for alpha =", alpha_val)
            print(spectrum_mix)

            spectrum_obs = Y[5]
            spectrum_pred = linear_mixing_model(soil, npv, alpha_val)
            
            spectrum_obs = np.array(spectrum_obs).flatten()
            spectrum_pred = np.array(spectrum_pred).flatten()

            
# Visualizations
# ---------------------------------------
# continuum_removal
    spectrum = np.array([0.5, 0.6, 0.55, 0.65, 0.7, 0.68, 0.72, 0.74, 0.73, 0.75])
    plot_spectrum_continuum_removal(spectrum)

    spectrum = np.array([1, 2, 1.5, 2.5, 3, 2.8, 3.2, 3.4, 3.3, 3.5])
    plot_spectrum_normalization(spectrum)
    
    spectrum = np.array([1, 2, 1.5, 2.5, 3, 2.8, 3.2, 3.4, 3.3, 3.5])
    plot_spectrum_derivative(spectrum)
    
    np.random.seed(42)
    spectra = np.random.rand(100, 10)
    pca_data = principle_componenet_analysis(spectra, n_components=5)
    plot_pca_2d(pca_data)
    pca_data_3 = principle_componenet_analysis(spectra, n_components=3)
    plot_pca_3d(pca_data_3)
    
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    model = create_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, epochs=50, batch_size=16)
    plot_loss(history)
    
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20, 1)
    model = create_model(X_train.shape[1])
    train_model(model, X_train, y_train, epochs=50, batch_size=16)
    validate_model(model, X_test, y_test)
    plot_predictions(model, X_test, y_test)
    
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    losses, maes = k_fold_cross_validation(X, y, k=5, epochs=50, batch_size=16)
    plot_kfold_results(losses, maes)
    
    accuracy(model, X_test, y_test)
    
    plot_spectral_curves(A, Y)
    plot_mixing_coefficients(a_estimated)
    plot_linear_mixing_model(soil, npv, alpha_val)
    plot_heatmaps(A, Y)
    
    pca_2d = principle_componenet_analysis(Y, n_components=2)
    pca_transformer_2d = PCA(n_components=2)
    pca_transformer_2d.fit(Y)
    endmembers_2d = pca_transformer_2d.transform(endmembers)
    plot_nfindr_2d(pca_2d, endmembers_2d)
    
    pca_3d = principle_componenet_analysis(Y, n_components=3)
    pca_transformer_3d = PCA(n_components=3)
    pca_transformer_3d.fit(Y)
    endmembers_3d = pca_transformer_3d.transform(endmembers)
    plot_nfindr_3d(pca_3d, endmembers_3d)