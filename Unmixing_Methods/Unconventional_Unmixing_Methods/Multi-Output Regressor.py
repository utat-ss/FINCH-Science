"""
This code is an implementation of multi-output regressor. The model will be specifically used for 
hyperspectral unmixing but it can be used for other purposes. To run the whole code, simply call
the run_rfmo_model() function with optional parameters defined below.
"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_rfmo_model(target_abundances=['gv','soil'], graph = True, n_estimators= 200, 
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = None,
    criterion = ['squared_error', 'absolute_error', 'friedman_mse']):

    #This function runs the whole multi-output regressor model with the specified target abundances (npv is always present)
    
    target_abundances = target_abundances + ['npv']
    target_cols = [abundance+'_fraction' for abundance in target_abundances]
    predict_columns = ['pred_' + abundance for abundance in target_abundances]

    x_train, x_test, y_train, y_test = ds_preparation(target_cols)

    pred_ds = multioutput_predict(x_train, y_train, x_test, predict_columns, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion)
    evaluate_model(pred_ds, y_test,predict_columns,target_abundances)
    if graph:
        graph_model(y_test,pred_ds,target_abundances,target_cols,predict_columns)


def ds_preparation(target_cols, file = '/Users/joshuasmacbookair/Desktop/UTAT/simpler_data.csv', wavelength_lim= [900,1700], test_size=0.2):   
    # This funciton imports defines the features and filters the data set. Put your data as a file path when you call this function
    
    # Load dataset 
    ds = pd.read_csv(file)

    # Filter wavelength columns default between 900 and 1700
    wavelength_cols = [col for col in ds.columns if col.isdigit() and wavelength_lim[0] <= int(col) <= wavelength_lim[1]]

    # Keep wavelength columns and target labels
    ds_filtered = ds[wavelength_cols + target_cols]
    
    # Define features (x) and target variables (y)
    x = ds_filtered[wavelength_cols]
    y = ds_filtered[target_cols]

    # Split the dataset into training and testing sets (80% train, 20% test, can change it)
    return train_test_split(x, y, test_size=test_size, random_state=42)

def multioutput_predict(x_train, y_train, x_test, predict_columns, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion):
    # This function creates a multi-output regressor model that is trained and tested

    multi_rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        )    
    multi_rf.fit(x_train, y_train)

    # Predict using a multi-output regression model
    predictions = multi_rf.predict(x_test)
    pred_ds = pd.DataFrame(predictions, columns=predict_columns)
    return pred_ds

def evaluate_model(predictions, labels, predict_columns, target_abundances):
    # This function evaluates the created model by looking at its mse and r2
    
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    r2_npv = r2_score(labels['npv_fraction'], predictions['pred_npv'])
    print(f"\nMean Squared Error: {mse}")
    if len(target_abundances)>1 :
        print(f"R-squared: {r2}")
    print (f"R-squared for npv only: {r2_npv}")

def graph_model(labels, predictions, target_abundances,target_cols,predict_columns):
    
    # Plotting Actual vs Predicted for each target variable
    plt.figure(figsize=(10, 6))
    for target_col, predict_col, abundance in zip(target_cols, predict_columns, target_abundances):
        plt.scatter(labels[target_col], predictions[predict_col], label=abundance, alpha=0.6)

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # y = x reference line
    plt.grid(True)
    plt.show()

    if len(target_abundances)>1 :
        # Plotting Actual vs Predicted for npv
        plt.figure(figsize=(10, 6))
        plt.scatter(labels['npv_fraction'], predictions['pred_npv'], label='NPV', alpha=0.6, color='orange')
        plt.xlabel("Actual NPV")
        plt.ylabel("Predicted NPV")
        plt.title("Actual vs Predicted NPV")
        plt.legend()
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.grid(True)
        plt.show()

run_rfmo_model()