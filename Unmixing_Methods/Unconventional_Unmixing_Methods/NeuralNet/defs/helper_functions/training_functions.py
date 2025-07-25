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

from data_manipulation_functions import K_Fold_Crossval_Data
from create_model_and_optimizer_functions import *


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

def train_k_fold(cfg_train, cfg_dataset, cfg_model, cfg_optim, cfg_plots, dataset, device, loss):

    """
    Trains various networks using k-fold cross-val, returns them all.

    Parameters
        - dataset (torch.tensor): The entire dataset to be used as a tensor, must include true indices and true spectral names since we are shuffling

        - cfg_train (dict): A dictionary that contains:
            - fold (int): How many folds we want
            - 

        - cfg_dataset (dict): A dictionary to process dataset that contains:
            - abundance_idx_range (list(int)): A list that specifies start-end indices of abundances
            - value_idx_range (list(int)): A list that specifies start-end indices of spectral values    
    """

    K_Folder = K_Fold_Crossval_Data(dataset= dataset, cfg_train= cfg_train, cfg_dataset= cfg_dataset)

    # Initialize the error metrics to handle model comparing
    error_metric_absolute = 1e5
    error_metric_list = []
    models_list = []

    for i in range(K_Folder.fold):
        
        # Get the data at that fold
        train_abundances, train_values, train_identities, val_abundaces, val_values, val_identities = K_Folder.get_data_at_fold(i)

        # Define the network
        model = initialize_model(cfg_model= cfg_model)
        model.to(device)
        model.train()
        
        # Initialize optimizer and scheduler, if passed
        optimizer, scheduler = initialize_optimizer(cfg_optim= cfg_optim)

        # Track train_losses and val_losses to 











        # train the model using the kth fold
        #    during this, output the graph of different parts of the loss function (reconstr spectra wise doesn't sound like a bad idea, and the physical loss)

        # validate the model on kth fold

        # based on validation results, get the error metric

        if interim_error_metric < error_metric:
            ...


        # if the error metric is less than the 


        ...
        
    ...