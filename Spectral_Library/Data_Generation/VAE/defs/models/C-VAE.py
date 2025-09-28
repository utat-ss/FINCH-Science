import json

import torch
import torch.nn as nn

# This file is used to define the Conditional - Variational Auto Encoder

class VAE(nn.Module):

    """
    This code is for a conditional variational autoenco
    """

    def __init__(self, cfg_cvae):
        super().__init__()

        

