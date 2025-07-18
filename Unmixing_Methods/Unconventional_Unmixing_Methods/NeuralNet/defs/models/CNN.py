import torch.nn as nn

class CNN1D_MLP(nn.Module):

    def __init__(self, cfg_CNNMLP: dict):
        super().__init__()

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
        self.hidden_conv_pooltype = cfg_CNNMLP.get('hidden_conv_pooltype', []) ###################
        self.hidden_conv_pool_kernelsize = cfg_CNNMLP.get('hidden_conv_pool_kernelsize', [])
        self.hidden_conv_pool_stride = cfg_CNNMLP.get('hidden_conv_pool_stride', [])

        # Activation functions
        activation_map = {"Linear_ReLU": nn.ReLU(), "Linear_Sigmoid": nn.Sigmoid(), "Linear_LeakyReLU": nn.LeakyReLU(), "Linear_ELU": nn.ELU(), "Linear_Tanh": nn.Tanh(), "Linear_SiLU": nn.SiLU()}
        self.linear_activation_list = cfg_CNNMLP['linear_activation_list']
        for i, key in enumerate(self.linear_activation_list): self.linear_activation_list[i] = activation_map[key]
        self.conv_activation_list = cfg_CNNMLP['conv_activation_list']
        for i, key in enumerate(self.conv_activation_list): self.conv_activation_list[i] = activation_map[key]

        # Construct the model, it has convolution layers first and then the MLP part
        cnn_layers = []
        sequence_length = self.i_dim # We have to keep track of how the signals' length changes as it goes through, we'll update this frequently

        assert len(self.hidden_conv_dim) == len(self.hidden_conv_kernelsize) == len(self.hidden_conv_stride) == len(self.hidden_conv_padding) == len(self.conv_activation_list), "All conv config lists must be same length (excluding pooling)"

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

        self.post_cnn_sequence_length = self.hidden_conv_dim[-1] * sequence_length


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

    def forward(self, inputs): #avoid bug if need to call input() for whatever reason in case
        return self.CNN1D_MLP(inputs)