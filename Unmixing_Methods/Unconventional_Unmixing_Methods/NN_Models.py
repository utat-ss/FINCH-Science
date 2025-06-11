"""
This is the file where we define the NN unmixing models,
we will run the code in the .ipynb file to test it.
"""

#region Required Imports

import torch.nn as nn
import torch
import torchvision

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

        self.activation_list = cfg_MLP['linear_activation_list']
        activation_map = {"ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "LeakyReLU": nn.LeakyReLU(), "ELU": nn.ELU(), "Tanh": nn.Tanh(), "SiLU": nn.SiLU()}
        for i, key in enumerate(self.activation_list): self.activation_list[i] = activation_map[key]

        # Define the MLP
        modules = []

        modules.append(nn.Linear(in_features=self.i_dim, out_features=self.hidden_layers[0]))
        modules.append(self.activation_list[0])

        assert len(self.activation_list) == (len(self.hidden_layers)+1), "Activation list must have 1 more element than hidden layers since it also includes activation of output layer"

        for i in range(len(self.hidden_layers)-1):
            modules.append(nn.Linear(in_features=self.hidden_layers[i], out_features=self.hidden_layers[i+1]))
            modules.append(self.activation_list[i+1])

        modules.append(nn.Linear(in_features=self.hidden_layers[-1], out_features=self.o_dim))
        modules.append(self.activation_list[-1])

        self.MLP = nn.Sequential(*modules)
    
    def forward(self, input):
        return self.MLP(input)


class CNN1D_MLP(nn.Module):

    def __init__(self, cfg_CNNMLP: dict):

        # Take in the necessary definitions
        # IO
        self.i_dim = cfg_CNNMLP['input_dim'] # The dimensions of the different kinds of inputs we give to it through various channels
        self.o_dim = cfg_CNNMLP['output_dim']
        self.input_channels = cfg_CNNMLP['input_channels'] # How many kind of inputs we are feeding at the same time, we can feed a combination of spectra and its derivatives at the same time

        # MLP layer specifics
        self.hidden_linear_dim = cfg_CNNMLP['hidden_linear_dim']

        # CNN layer specifics
        self.hidden_conv_dim = cfg_CNNMLP['hidden_conv_dim']
        self.hidden_conv_kernelsize = cfg_CNNMLP.get('hidden_conv_kernelsize', [3] * len(self.hidden_conv_dim)) # Gets a default of 3
        self.hidden_conv_stride = cfg_CNNMLP.get('hidden_conv_stride', [1] * len(self.hidden_conv_dim)) # Gets a default of 1
        self.hidden_conv_padding = cfg_CNNMLP.get('hidden_conv_padding', [k // 2 for k in self.hidden_conv_kernelsize]) # Gets a default of 'same' padding
        self.hidden_conv_pooltype = cfg_CNNMLP.get('hidden_conb_pool_type', None)
        self.hidden_conv_pool_kernelsize = cfg_CNNMLP.get('hidden_conv_pool_kernelsize', None)
        self.hidden_conv_pool_stride = cfg_CNNMLP.get('hidden_conv_pool_stride', None)

        # Activation functions
        activation_map = {"Linear_ReLU": nn.ReLU(), "Linear_Sigmoid": nn.Sigmoid(), "Linear_LeakyReLU": nn.LeakyReLU(), "Linear_ELU": nn.ELU(), "Linear_Tanh": nn.Tanh(), "Linear_SiLU": nn.SiLU()}
        self.linear_activation_list = cfg_CNNMLP['linear_activation_list']
        for i, key in enumerate(self.linear_activation_list): self.linear_activation_list[i] = activation_map[key]
        self.conv_activation_list = cfg_CNNMLP['linear_activation_list']
        for i, key in enumerate(self.conv_activation_list): self.conv_activation_list[i] = activation_map[key]

        # Construct the model, it has convolution layers first and then the MLP part
        cnn_layers = []
        sequence_length = self.i_dim # We have to keep track of how the signals' length changes as it goes through, we'll update this frequently

        assert len(self.hidden_conv_dim) == len(self.hidden_conv_kernelsize) == len(self.hidden_conv_stride) == len(self.hidden_conv_padding) == len(self.conv_activation_list), "All conv config lists must be same length (excluding pooling)"
        assert len(self.conv_activation_list) == (len(self.hidden_conv_dim)+1), ""

        # Initial Conv Layer
        cnn_layers.append(nn.Conv1d(in_channels=self.input_channels, out_channels=self.hidden_conv_dim[0], kernel_size=self.hidden_conv_kernelsize[0], stride=self.hidden_conv_stride[0], padding=self.hidden_conv_padding[0])) 
        cnn_layers.append(self.conv_activation_list[0])

        sequence_length = (sequence_length + 2 * self.hidden_conv_padding[0] - self.hidden_conv_kernelsize[0]) // self.hidden_conv_stride[0] + 1 # Update length after convolution layer

        assert len(self.hidden_conv_pool_stride) == len(self.hidden_conv_pool_kernelsize) == len(self.hidden_conv_pooltype), "Pool lengths must be equal"

        if self.hidden_conv_pooltype is not None:

            if self.hidden_conv_pooltype[0] == "max":
                cnn_layers.append(nn.MaxPool1d(kernel_size=self.hidden_conv_pool_kernelsize[0], stride=self.hidden_conv_pool_stride[0]))
            elif self.hidden_conv_pooltype[0] == "avg":
                cnn_layers.append(nn.AvgPool1d(kernel_size=self.hidden_conv_pool_kernelsize[0], stride=self.hidden_conv_pool_stride[0]))
            else:
                raise ValueError(f"Unsupported pool type: {self.hidden_conv_pooltype[0]}")
            
            sequence_length = (sequence_length - self.hidden_conv_pool_kernelsize[0]) // self.hidden_conv_pool_stride[0] + 1 # Update length after pool layer

        # Repeating Conv Layers
        for i in range(len(self.hidden_conv_dim)-1):
            cnn_layers.append(nn.Conv1d(in_channels=self.hidden_conv_dim[i], out_channels=self.hidden_conv_dim[i+1], kernel_size=self.hidden_conv_kernelsize[i+1], stride=self.hidden_conv_stride[i+1], padding=self.hidden_conv_padding[i+1]))
            cnn_layers.append(self.conv_activation_list[i+1])
            
            sequence_length = (sequence_length + 2 * self.hidden_conv_padding[i+1] - self.hidden_conv_kernelsize[i+1]) // self.hidden_conv_stride[i+1] + 1 # Update length after convolution layer

            if self.hidden_conv_pooltype is not None:
                if self.hidden_conv_pooltype[i+1] == "max":
                    cnn_layers.append(nn.MaxPool1d(kernel_size=self.hidden_conv_pool_kernelsize[i+1], stride=self.hidden_conv_pool_stride[i+1]))
                elif self.hidden_conv_pooltype[i+1] == "avg":
                    cnn_layers.append(nn.AvgPool1d(kernel_size=self.hidden_conv_pool_kernelsize[i+1], stride=self.hidden_conv_pool_stride[i+1]))
                else:
                    raise ValueError(f"Unsupported pool type: {self.hidden_conv_pooltype[i+1]}")
            
                sequence_length = (sequence_length - self.hidden_conv_pool_kernelsize[i+1]) // self.hidden_conv_pool_stride[i+1] + 1 # Update length after pool layer

        self.post_cnn_sequence_length = self.i_dim * sequence_length

        # Initial MLP Layer
        mlp_layers = []

        mlp_layers.append(nn.Linear(in_features=self.post_cnn_sequence_length, out_features=self.hidden_linear_dim[0]))
        mlp_layers.append(self.linear_activation_list[0])

        for i in range(len(self.hidden_linear_dim)-1):
            mlp_layers.append(nn.Linear(in_features=self.hidden_linear_dim[i], out_features=self.hidden_linear_dim[i+1]))
            mlp_layers.append(self.linear_activation_list[i+1])

        mlp_layers.append(nn.Linear(in_features=self.hidden_linear_dim[-1], out_features=self.o_dim))
        mlp_layers.append(self.linear_activation_list[-1])

        self.CNN1D_MLP = nn.Sequential(*cnn_layers, nn.Flatten(), *mlp_layers)

    def forward(self, input):
        return self.CNN1D_MLP(input)


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

