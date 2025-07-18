import json

import torch
import torch.nn as nn

#region NIF Models

class NIF_Pointwise(nn.Module):

    # This NIF model is a redifinition of the older one, here the latent space is parsed and then each parse is pointwise multiplied with the hidden layers pre-activation

    def __init__(self, cfg_shape_net, cfg_param_net):
        super(NIF_Pointwise, self).__init__()

        # Initialize shape network parameters
        self.cfg_shape_net = cfg_shape_net
        self.shape_i_dim = cfg_shape_net['input_dim']
        self.shape_o_dim = cfg_shape_net['output_dim']
        self.shape_hidden_units = cfg_shape_net['hidden_units']
        self.shape_activation = cfg_shape_net.get('shape_activation', nn.Tanh)()

        # Initialize paremeter network parameters
        self.cfg_param_net = cfg_param_net
        self.param_i_dim = cfg_param_net['input_dim']
        self.param_hidden_units = cfg_param_net['hidden_units']
        self.param_activation = cfg_param_net.get('param_activation', nn.Tanh)()

        self.param_latent_dim = sum(self.shape_hidden_units) + self.shape_o_dim # Define length of latent dim of param net


        # -----Build param network-----

        """
        Initializes the parameter net. The latent space of parameter net will feed into the shape net's last layer's weights.
        """

        # Add input layer
        self.p_layers = [nn.Linear(in_features=self.param_i_dim, out_features=self.param_hidden_units[0]), self.param_activation]

        # Add hidden layers
        for i in range(len(self.param_hidden_units)-1):
            self.p_layers.append(nn.Linear(in_features=self.param_hidden_units[i], out_features=self.param_hidden_units[i+1]))
            self.p_layers.append(self.param_activation)

        # Add output (latent space) layer
        self.p_layers.append(nn.Linear(in_features=self.param_hidden_units[-1], out_features=self.param_latent_dim))
        self.p_layers.append(self.param_activation)

        self.parameter_network = nn.Sequential(*self.p_layers)


        # -----Build shape network-----

        """
        Initializes the shape net. The latent space of parameter net will feed into the shape net's last layer's weights.
        
        """

        self.shape_network = nn.ModuleList()

        # Add input layer
        self.shape_network.append(nn.Linear(in_features=self.shape_i_dim, out_features=self.shape_hidden_units[0]))

        # Add hidden layers
        for i in range(len(self.shape_hidden_units)-1):
            self.shape_network.append(nn.Linear(in_features=self.shape_hidden_units[i], out_features=self.shape_hidden_units[i+1]))

        # Add output layer
        self.shape_network.append(nn.Linear(in_features=self.shape_hidden_units[-1], out_features=self.shape_o_dim))

    def _call_shape_network(self, shape_input, latent_space):

        """
        This is a little untraditional... Hwere we call each layer normally, but at the end of each layer before applying 
        an activation function, we pointwise multiply with some part of the latent space vector obtained from param net.
        Then we obviously input this into an activation function.
        """

        start_offset = 0 # to make sure we start slicing from start and then have a moving slicer

        self.total_units = self.shape_hidden_units + [self.shape_o_dim] # add the output dim since we have as many outputs as the hidden units + output layer

        for layer, out_dim in zip(self.shape_network, self.total_units):

            out = layer(shape_input) # take the result from layer

            latent_slice = latent_space[:, start_offset:start_offset+out_dim] # slice a part of latent space
            out = out * latent_slice # point wise multiply the out and latent space

            shape_input = self.shape_activation(out) # put into the activation function
            start_offset += out_dim # offset the slicing start

        return shape_input # return the output once done

    def forward(self, inputs):

        shape_input = inputs[:, :self.shape_i_dim]
        param_input = inputs[:, self.shape_i_dim:]

        latent_space = self.parameter_network(param_input) # take the latent space from parameter network

        return self._call_shape_network(shape_input, latent_space) # call the shape network
    
    def save_config(self, filename_config="config.json"):
        """
        Saves the NIF model configuration to a JSON file.

        Args:
            filename (str, optional): The name of the file to save the
            configuration. Defaults to "config.json".
        """
        config = {
            "cfg_shape_net": self.cfg_shape_net,
            "cfg_parameter_net": self.cfg_param_net,
        }
        with open(filename_config, "w") as write_file:
            json.dump(config, write_file, indent=4)


class NIF_PartialPaper(nn.Module):

    # Here, we define another custom version of the Paper NIF where we do not initialize the entirety of the shape network but only a part of it

    def __init__(self, cfg_shape_net, cfg_param_net):
        super(NIF_PartialPaper, self).__init__()

        # Initialize shape network parameters
        self.cfg_shape_net = cfg_shape_net
        self.shape_i_dim = cfg_shape_net['input_dim']
        self.shape_o_dim = cfg_shape_net['output_dim']
        self.shape_hidden_units = cfg_shape_net['hidden_units']
        self.shape_activation = cfg_shape_net.get('shape_activation', nn.Tanh)()

        # Initialize paremeter network parameters
        self.cfg_param_net = cfg_param_net
        self.param_i_dim = cfg_param_net['input_dim']
        self.param_generated_unit = cfg_param_net['generated_unit']
        self.param_hidden_units = cfg_param_net['hidden_units']
        self.param_activation = cfg_param_net.get('param_activation', nn.Tanh)()


        # -----Build param network-----

        """
        Initializes the parameter net. The latent space of parameter net will feed into the shape net's last layer's weights.
        """

        # We first have to calcuate the shape of the latent space, based on what hidden layers are input
        self.param_latent_dim = (self.shape_i_dim * self.param_generated_unit) + self.param_generated_unit

        # Add input layer
        self.p_layers = [nn.Linear(in_features=self.param_i_dim, out_features=self.param_hidden_units[0]), self.param_activation]

        # Add hidden layers
        for i in range(len(self.param_hidden_units)-1):
            self.p_layers.append(nn.Linear(in_features=self.param_hidden_units[i], out_features=self.param_hidden_units[i+1]))
            self.p_layers.append(self.param_activation)

        # Add output (latent space) layer
        self.p_layers.append(nn.Linear(in_features=self.param_hidden_units[-1], out_features=self.param_latent_dim))
        self.p_layers.append(self.param_activation)

        self.parameter_network = nn.Sequential(*self.p_layers)


        # -----Build shape network-----

        """
        Initializes the shape net without the generated layer. The latent space of parameter net will feed into the shape net's last layer's weights.
        
        """

        self.shape_network_no_generated = nn.ModuleList()

        # Add input layer
        self.shape_network_no_generated.append(nn.Linear(in_features=self.param_generated_unit, out_features=self.shape_hidden_units[0])) # Generated layer and hidden layers connection

        # Add hidden layers
        for i in range(len(self.shape_hidden_units)-1): 
            self.shape_network_no_generated.append(nn.Linear(in_features=self.shape_hidden_units[i], out_features=self.shape_hidden_units[i+1]))

        # Add output layer
        self.shape_network_no_generated.append(nn.Linear(in_features=self.shape_hidden_units[-1], out_features=self.shape_o_dim))

    def _call_shape_network(self, shape_input, latent_space):

        """
        This is a little untraditional... Here we call each layer normally, but the initial layer is completely defined by the latent space.
        """

        assert latent_space.shape[0] == shape_input.shape[0], "Latent space and shape input must have the same batch size."

        assert latent_space.shape[1] == self.param_latent_dim, f"Latent space must have {self.param_latent_dim} dimensions, but got {latent_space.shape[1]}." 

        # We first have to slice the latent space to get the defined layer

        generated_W = (latent_space[:,:self.param_generated_unit * self.shape_i_dim]).view(latent_space.shape[0], self.param_generated_unit, self.shape_i_dim)
        generated_b = latent_space[:,self.param_generated_unit * self.shape_i_dim:]

        # Unsqueeze it into (B, input_dim, 1) to do batch matrix multiplication
        shape_input = shape_input.unsqueeze(2) 
        
        # Do matmul: (B, generated_unit, input_dim) x (B, input_dim, 1) -> (B, generated_unit, 1), squeeze it to (B, generated_unit), add bias, use activation function
        out = self.shape_activation((torch.bmm(generated_W, shape_input)).squeeze(2) + generated_b)

        for layer in self.shape_network_no_generated:

            out = layer(out) # take the result from layer
            out = self.shape_activation(out) # put into the activation function

        return out # return the output once done

    def forward(self, inputs):

        shape_input = inputs[:, :self.shape_i_dim]
        param_input = inputs[:, self.shape_i_dim:]

        latent_space = self.parameter_network(param_input) # take the latent space from parameter network

        return self._call_shape_network(shape_input, latent_space) # call the shape network
    
    def save_config(self, filename_config="config.json"):
        """
        Saves the NIF model configuration to a JSON file.

        Args:
            filename (str, optional): The name of the file to save the
            configuration. Defaults to "config.json".
        """
        config = {
            "cfg_shape_net": self.cfg_shape_net,
            "cfg_parameter_net": self.cfg_param_net,
        }
        with open(filename_config, "w") as write_file:
            json.dump(config, write_file, indent=4)