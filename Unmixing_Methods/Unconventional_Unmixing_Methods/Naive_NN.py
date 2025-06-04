"""
This is the file where we define the Naive NN unmixing method,
we will run the code in the .ipynb file to test it.
"""

#region Required Imports

import torch.nn as nn
import torch

#endregion


class NaiveNN(nn.Module):

    def __init__(self, cfg_MLP: dict):

        self.i_dim = cfg_MLP['input_dim']
        self.o_dim = cfg_MLP['output_dim']
        self.hidden_layers = cfg_MLP['hidden_layers']

        self.activation_function = cfg_MLP.get(nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.SiLU)

    def _initialize_MLP(self):

        modules = []

        modules.append(nn.Linear(in_features=self.i_dim, out_features=self.hidden_layers[0]))
        modules.append(self.activation_function)

        for i in range(len(self.hidden_layers)):
            modules.append(nn.Linear(in_features=self.hidden_layers[i], out_features=self.hidden_layers[i+1]))
            modules.append(self.activation_function)

        modules.append(nn.Linear(in_features=self.hidden_layers[-1], out_features=self.o_dim))
        modules.append(self.activation_function)

        return nn.Sequential(*modules)