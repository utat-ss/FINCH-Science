import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, cfg_MLP: dict):
        super().__init__()

        # Take in the necessary definitions
        self.i_dim = cfg_MLP['input_dim']
        self.o_dim = cfg_MLP['output_dim']
        self.hidden_layers = cfg_MLP['hidden_layers']

        self.activation_list = cfg_MLP['linear_activation_list']
        activation_map = {"ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "LeakyReLU": nn.LeakyReLU(), "ELU": nn.ELU(), "Tanh": nn.Tanh(), "SiLU": nn.SiLU(),"Identity": nn.Identity()}
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