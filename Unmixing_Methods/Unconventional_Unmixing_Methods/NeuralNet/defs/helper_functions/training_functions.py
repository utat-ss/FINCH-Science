from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

import sys
from pathlib import Path

github_root = Path(__file__).resolve().parents[2]
sys.path.append(str(github_root))

from defs.models.CNN import *
from defs.models.MLP import *
from defs.models.NIF import *

from defs.helper_functions.data_manipulation_functions import K_Fold_Crossval_Data
from defs.helper_functions.create_model_and_optimizer_functions import *


def train_Network(cfg_dataset, cfg_train, cfg_plots, data_array: np.ndarray, device, model, optimizer, scheduler= None):

    X = data_array[:, cfg_dataset['idx_range_tuple'][0] : cfg_dataset['idx_range_tuple'][1]]
    Y = data_array[:, cfg_dataset['idx_ab_tuple'][0]   : cfg_dataset['idx_ab_tuple'][1]]
    
    #Shuffle
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n = len(X)
    t0, t1 = cfg_train['seperation_ratios'][0], cfg_train['seperation_ratios'][1]
    i0, i1 = int(n*t0), int(n*t1)
    X_train,  Y_train  = X[:i0],  Y[:i0]
    X_val,    Y_val    = X[i0:i1], Y[i0:i1]
    X_test,   Y_test   = X[i1:],   Y[i1:]

    #DataLoaders
    batch_size = cfg_train['batch_size']
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_dataset   = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    
    test_dataset       = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
    test_loader   = DataLoader(test_dataset, batch_size=batch_size)

    #Loss
    criterion = nn.MSELoss()


    num_epochs = cfg_train.get('num_epochs', 20)
    history = {'train_loss': [], 'val_loss': []}

    # training n validation
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation section
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            test_loss += criterion(model(xb), yb).item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"\nFinal test loss: {test_loss:.4f}")

    return model, history, test_loss

def train_k_fold(cfg_train, cfg_dataset, cfg_model, cfg_optim, dataset, device, loss):

    """
    Trains various networks using k-fold cross-val, returns them all.

    Parameters
        - dataset (torch.tensor): The entire dataset to be used as a tensor, must include true indices and true spectral names since we are shuffling

        - cfg_train (dict): A dictionary that contains:
            - fold (int): How many folds we want
            - batch_size (int): Batch size, default is 4

        - cfg_dataset (dict): A dictionary to process dataset that contains:
            - abundance_idx_range (list(int)): A list that specifies start-end(included) indices of abundances
            - value_idx_range (list(int)): A list that specifies start-end(included) indices of spectral values
            - identity_idx_range (list(int)): A list that specifies start-end(included) indice(s) of name, and original index
            - property_idx_range (list(int)): A list that specifies start-end(included) indice(s) of additional properties such as RWC, VZA, etc.
            - seed (int): For replicibility of shuffle, def=69

        - cfg_model (dict): Look at the specific model class you are using for this

        - cfg_optim (dict): Look at the optimizer (and scheduler if applicable) generation functions for this

    Returns
        - folds_dict (dict): It has:
            - for each fold_{i}_dict (dict):
                - fold_loss_dict (dict): A dict that has all kinds of losses for both train and val:
                    - total_loss
                    - reconstruction
                    - sum
                    - bounds
                - fold_predictions_dict: dicts for abundance predictions for both train and val
                - trained_model: model.state_dict() so that we can reuse it
    """

    folds_dict = {}

    # Initialize the folder for all the specific folds
    K_Folder = K_Fold_Crossval_Data(dataset= dataset, cfg_train= cfg_train, cfg_dataset= cfg_dataset)

    # Repeat this for each fold
    for i in range(K_Folder.fold):
        
        # Get the data at that fold and load the data
        train_abundances, train_values, train_identities, val_abundances, val_values, val_identities = K_Folder.get_data_at_fold(i)

        # Load the training dataset, do not shuffle since we have already did
        train_loader = DataLoader(
            TensorDataset(train_values, train_abundances), 
            batch_size= cfg_train.get('batch_size', 4), 
            shuffle= False
            )

        # Load the validation dataset, do not shuffle since we have already did
        val_loader = DataLoader(
            TensorDataset(val_values, val_abundances), 
            batch_size= 1, # Batch is 1 cuz we want to see individual samples' behavior
            shuffle= False
            ) 
 
        # Dict where we will record the losses of this fold
        fold_loss_dict = {
            "train": {
                "total": [],
                "reconstruction": [],
                "sum": [],
                "bounds": [],
            },
            "val": {
                "total": [],
                "reconstruction": [],
                "sum": [],
                "bounds": []
            }
        }

        # Dict where we will record the prediction arrays of this fold
        fold_predictions_dict = {
            "train": [],
            "val": []
        }

        # Define the network, set to training mode and send to device
        model = initialize_model(cfg_model= cfg_model, device= device)
        model.to(device)
        model.train()

        # Initialize optimizer and scheduler (if passed)
        optimizer, scheduler = initialize_optimizer(cfg_optim= cfg_optim, model= model)

        # Train for all batches
        idx =1
        for train_inputs, train_outputs in train_loader:

            # Send data to the same device as model
            train_inputs, train_outputs = train_inputs.to(device), train_outputs.to(device)

            # Reset grad, get predictions
            optimizer.zero_grad()
            pred = model(train_inputs)

            # Calculate loss
            total_loss, reconstruction_loss, sum_loss, bounds_loss = loss(
                pred= pred, target= train_outputs
            )

            # Take loss, backpropogate, and update
            total_loss.backward()
            optimizer.step()

            # Append the losses to the lists
            fold_loss_dict['train']['total'].append(total_loss.item())
            fold_loss_dict['train']['reconstruction'].append(reconstruction_loss.detach().cpu().numpy())
            fold_loss_dict['train']['sum'].append(sum_loss.item())
            fold_loss_dict['train']['bounds'].append(bounds_loss.item())

            # Convert the preds into cpu so that we can make a numpy array of them
            pred_cpu = pred.detach().cpu().numpy()
            fold_predictions_dict['train'].append(pred_cpu)

            print(f'Training {idx} at fold {i} finished')
            idx +=1

        # Setto evaluation mode
        model.eval()

        # We do not need grads for evals since we won't backprop
        with torch.no_grad():

            # For all value at the validation datasets
            idx =1
            for val_inputs, val_outputs in val_loader:

                # Send the data to the same device as model
                val_inputs, val_outputs = val_inputs.to(device), val_outputs.to(device)

                # Get predictions
                pred = model(val_inputs)

                # Calculate loss
                total_loss, reconstruction_loss, sum_loss, bounds_loss = loss(
                    pred= pred, target= val_outputs
                )

                # Append the losses to the lists
                fold_loss_dict['val']['total'].append(total_loss.item())
                fold_loss_dict['val']['reconstruction'].append(reconstruction_loss.detach().cpu().numpy())
                fold_loss_dict['val']['sum'].append(sum_loss.item())
                fold_loss_dict['val']['bounds'].append(bounds_loss.item())

                # Convert the preds into cpu so that we can make a numpy array of them
                pred_cpu = pred.detach().cpu().numpy()
                fold_predictions_dict['val'].append(pred_cpu)
                
                print(f'Validation {idx} at fold {i} finished')
                idx +=1

        # Compile all the dicts to a master dict
        folds_dict[f'fold_{i}_dict'] = {}
        folds_dict[f'fold_{i}_dict']['fold_loss_dict'] = fold_loss_dict
        folds_dict[f'fold_{i}_dict']['fold_predictions_dict'] = fold_predictions_dict
        folds_dict[f'fold_{i}_dict']['trained_model'] = model.state_dict()

    return folds_dict
