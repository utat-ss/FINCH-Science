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

def train_Network(cfg_NN, cfg_dataset, cfg_train, cfg_plots, data_array: np.ndarray, device, model):

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
    
    model.to(device)

    #Loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg_train.get('lr', 1e-4))

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

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            test_loss += criterion(model(xb), yb).item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"\nFinal test loss: {test_loss:.4f}")

    return model, history, test_loss

