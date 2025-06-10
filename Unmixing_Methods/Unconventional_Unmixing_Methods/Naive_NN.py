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

        activation_name = cfg_MLP.get("ReLU", "Sigmoid", "LeakyReLU", "ELU", "Tanh", "SiLU")
        activation_map = {"ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "LeakyReLU": nn.LeakyReLU(), "ELU": nn.ELU(), "Tanh": nn.Tanh(), "SiLU": nn.SiLU()}
        self.activation_function = activation_map.get(activation_name, nn.ReLU)

        # Define the MLP
        modules = []

        modules.append(nn.Linear(in_features=self.i_dim, out_features=self.hidden_layers[0]))
        modules.append(self.activation_function)

        for i in range(len(self.hidden_layers)-1):
            modules.append(nn.Linear(in_features=self.hidden_layers[i], out_features=self.hidden_layers[i+1]))
            modules.append(self.activation_function)

        modules.append(nn.Linear(in_features=self.hidden_layers[-1], out_features=self.o_dim))
        modules.append(self.activation_function)

        self.MLP = nn.Sequential(*modules)
    
    def forward(self, input):
        return self.MLP(input)
    
class CNN1D(nn.Module):

    def __init__(self, cfg_ConvMLP: dict):

        # Take in the necessary definitions
        self.i_dim = cfg_ConvMLP['input_dim']
        self.o_dim = cfg_ConvMLP['output_dim']
        self.hidden_linear_dim = cfg_ConvMLP['hidden_linear_dim']

        self.hidden_conv_dim = cfg_ConvMLP['hidden_conv_dim']
        self.hidden_conv_kernelsize = cfg_ConvMLP['hidden_conv_kernelsize']
        self.hidden_conv_stride = cfg_ConvMLP['hidden_conv_stride']
        self.hidden_conv_padding = cfg_ConvMLP['hidden_conv_padding']
        self.hidden_conv_maxpool_kernelsize = cfg_ConvMLP['hidden_conv_maxpool_kernelsize']
        self.hidden_conv_maxpool_stride = cfg_ConvMLP['hidden_conv_maxpool_stride']

        activation_map = {"Linear_ReLU": nn.ReLU(), "Linear_Sigmoid": nn.Sigmoid(), "Linear_LeakyReLU": nn.LeakyReLU(), "Linear_ELU": nn.ELU(), "Linear_Tanh": nn.Tanh(), "Linear_SiLU": nn.SiLU()}
        linear_activation_name = cfg_ConvMLP.get("Linear_ReLU", "Linear_Sigmoid", "Linear_LeakyReLU", "Linear_ELU", "Linear_Tanh", "Linear_SiLU")
        self.linear_activation_function = activation_map.get(linear_activation_name, nn.ReLU)
        conv_activation_name = cfg_ConvMLP.get("Conv_ReLU", "Conv_Sigmoid", "Conv_LeakyReLU", "Conv_ELU", "Conv_Tanh", "Conv_SiLU")
        self.conv_activation_function = activation_map.get(conv_activation_name, nn.ReLU)

        # Now we construct the model the model has convolution layers first and then the MLP part
        modules = []

        assert len(self.hidden_conv_dim) == len(self.hidden_conv_kernelsize) == len(self.hidden_conv_stride) == len(self.hidden_conv_maxpool_stride) == len(self.hidden_conv_maxpool_kernelsize), "All conv config lists must be same length"
        
        # Initial Conv Layer
        modules.append(nn.Conv1d(in_channels=self.i_dim, out_channels=self.hidden_conv_dim[0], kernel_size=self.hidden_conv_kernelsize[0], stride=self.hidden_conv_stride[0], padding=self.hidden_conv_padding[0]))
        modules.append(self.conv_activation_function)

        # Now repeating Conv Layers

        if len(self.hidden_conv_dim) != 1:
            None




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



