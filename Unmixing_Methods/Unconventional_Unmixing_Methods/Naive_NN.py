"""
This is the file where we define the Naive NN unmixing method,
we will run the code in the .ipynb file to test it.
"""

#region Required Imports

import torch.nn as nn
import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#endregion


class MLP(nn.Module):

    def __init__(self, cfg_MLP: dict):
        super().__init__()

        # Take in the necessary definitions
        self.i_dim = cfg_MLP['input_dim']
        self.o_dim = cfg_MLP['output_dim']
        self.hidden_layers = cfg_MLP['hidden_layers']

        self.activation_function = cfg_MLP.get(nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.SiLU)

        # Define the MLP
        modules = []

        modules.append(nn.Linear(in_features=self.i_dim, out_features=self.hidden_layers[0]))
        modules.append(self.activation_function)

        for i in range(len(self.hidden_layers)):
            modules.append(nn.Linear(in_features=self.hidden_layers[i], out_features=self.hidden_layers[i+1]))
            modules.append(self.activation_function)

        modules.append(nn.Linear(in_features=self.hidden_layers[-1], out_features=self.o_dim))
        modules.append(self.activation_function)

        self.MLP = nn.Sequential(*modules)

        return self.MLP
    
    def _call_MLP(self, input):

        output = self.MLP(input)
        
        return output

def train_Network(cfg_NN: dict, cfg_dataset: dict, cfg_train: dict, cfg_plots: dict, input: np.array):

    # First, reshape the dataset to get classification and training data
    idx_data_trueab_tuple = cfg_dataset['idx_ab_tuple'] # a tuple definin the indices of the training dataset
    idx_data_range_tuple = cfg_dataset['idx_range_tuple'] # a tuple definition of the training dataset

    # Second, define how the training process is going to occur
    train_regularizer = cfg_train['regularizer'] # NFIY, regularizer for training
    batch_size = cfg_train['batch_size'] # Self explanatory, batch size. After each batch, validation runs will occur.
    seperation_ratios = ['seperation_ratios'] # a tuple defining ratios of seperation (%training, %validation, %test)

    # Third, define the plotting configs
    plot_losses = cfg_plots['plot_losses']
    plot_errors = cfg_plots['plot_errors']
    plot_seperately = cfg_plots['plot seperately']

    # Lastly, retrieve the type of NN
    model_type = cfg_NN['model_type']


    # We now preprocess the given dataset.
    # First step is to randomly sample from the input dataset and seperate it into training, validation, testing

    np.random.shuffle(input)

    num_rows = input.shape[0]

    data_train = input[:round(num_rows*seperation_ratios[0]), idx_data_range_tuple[0]:idx_data_range_tuple[1]]
    data_train_classification = input[:round(num_rows*seperation_ratios[0]), idx_data_trueab_tuple[0]:idx_data_trueab_tuple[1]]

    data_validate = input[round(num_rows*seperation_ratios[0]):round(num_rows*seperation_ratios[1]), idx_data_range_tuple[0]:idx_data_range_tuple[1]]
    data_validate_classification = input[round(num_rows*seperation_ratios[0]):round(num_rows*seperation_ratios[1]), idx_data_trueab_tuple[0]:idx_data_trueab_tuple[1]]

    data_test = input[round(num_rows*seperation_ratios[1]):, idx_data_range_tuple[0]:idx_data_range_tuple[1]]
    data_test_classification = input[round(num_rows*seperation_ratios[1]):, idx_data_trueab_tuple[0]:idx_data_trueab_tuple[1]]


    # We now have to initialize the model

    if model_type == 'MLP':

        model = MLP(cfg_NN)

    # We now train the model and plot at the same time

    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)

    counter = 0

    for i in range((input.shape[0]) - (data_test.shape[0])):

        optimizer.zero_grad()



