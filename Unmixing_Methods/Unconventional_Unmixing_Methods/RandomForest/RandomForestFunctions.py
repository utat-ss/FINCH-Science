#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
Author: Zoe, UTAT -SS - Science 

This code uses Random forest for the unmixing process. Sources: https://www.geeksforgeeks.org/random-forest-regression-in-python/, https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/

WARNING: TAKES SUPER LONG. This code includes hyperparameter tuning, which means that it can take like 20-25 minutes to tun. If you don't want to wait this long, get rid of the hyperparameter tuning section and instead just pick them (or take the default ones)

With simpler data only, Mean Squared Error: 0.026416158563140196 R-squared: 0.7257821483336598

More information on random forest on the notion: https://www.notion.so/utat-ss/FAE-Random-Forest-1b63e028b0ea80e3afcad34492232512

Running example: run_rf_model (target_abundances=['gv', 'soil'], graph = True, n_estimators= [200], criterion = ['squared_error'])

If you have any questions, feel free to ask me!
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def run_rf_model (target_abundances=['gv','soil'], graph = True, n_estimators= [200, 300], 
    max_depth= [None],
    min_samples_split= [2],
    min_samples_leaf= [1],
    max_features =[None],
    criterion = ['squared_error', 'absolute_error', 'friedman_mse']):

    #This function runs the whole Random Forest model with the specified target abundances (npv is always present) and hyperparameters. You also have the option of graphing the true vs. predicted plots.
    
    target_abundances.append('npv')
    target_cols = [abundance+'_fraction' for abundance in target_abundances]
    predict_columns = ['pred_' + abundance for abundance in target_abundances]

    x_train, x_test, y_train, y_test = ds_preparation(target_cols)
    pred_ds = grid_search_predict(x_train, y_train, x_test, predict_columns, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion)
    evaluate_model(pred_ds, y_test,predict_columns,target_abundances)
    if graph:
        graph_model(y_test,pred_ds,target_abundances,target_cols,predict_columns)


def ds_preparation(target_cols, file = '/Users/zoe/Downloads/simpler_data.csv', wavelength_lim= [900,1700], test_size=0.2):   
    # This funciton imports defines the features and filters the data set
    
    # Load dataset 
    ds = pd.read_csv(file) #put your data here!

    # Filter wavelength columns default between 900 and 1700
    wavelength_cols = [col for col in ds.columns if col.isdigit() and wavelength_lim[0] <= int(col) <= wavelength_lim[1]]

    # Keep wavelength columns and target labels
    ds_filtered = ds[wavelength_cols + target_cols]
    
    # Define features (x) and target variables (y)
    x = ds_filtered[wavelength_cols]
    y = ds_filtered[target_cols]

    # Split the dataset into training and testing sets (80% train, 20% test, can change it)
    return train_test_split(x, y, test_size=test_size, random_state=42)

def grid_search_predict(x_train, y_train, x_test, predict_columns, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion):
    #this function runs RandomForestRegressor for all of the combinations of the specified hyperparameters (will take a long time if you add many)
    
    param_grid = {
    'n_estimators': n_estimators, 
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
    'criterion': criterion
    }


    # trying the hyperparameters combinations
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3,  # how many cross-validations
                           n_jobs=-1,  # Use all available CPU cores
                           verbose=0)  #otherwise it prints everything

    # Fit the grid search on the training data
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters found 
    print("\nBest parameters found by GridSearchCV:")
    print(grid_search.best_params_)

    # Get the best model 
    best_model = grid_search.best_estimator_

    # Predict using the best model
    predictions= best_model.predict(x_test)
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

    #for index in range(len(target_abundances)):
    #plt.scatter(labels[target_cols[index]], predictions[predict_columns[index]], label=target_abundances[index], alpha=0.6)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # diagonal line for reference
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

