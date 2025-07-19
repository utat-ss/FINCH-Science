"""

This file is to define functions where we:

- Generate the model given a config
- Generate an optimizer given a config

"""

import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR

import sys
from pathlib import Path

github_root = Path(__file__).resolve().parents[2]
sys.path.append(str(github_root))

from defs.models.CNN import *
from defs.models.MLP import *
from defs.models.NIF import *

# Define the optimizer functions here, we'll use these to initialize the optimizers.

def initialize_optimizer(cfg_optim: dict, model: torch.nn.Module):

    """
    
    Initializes an optimizer to be used during training:

    cfg_optim parameters:
    - cfg_optim[algo] (str): which optimizer algorithm to be used, options are: "adamw" (default), "sgd"
    - cfg_optim[lr] (float): what (initial) learning rate must be, default == 1e-4
    - cfg_optim[lr_decay] (str): decay scheduer, options are: "none" (default), "exp", "step"
    """

    algo = cfg_optim.get('algo', 'adamw')
    lr_decay = cfg_optim.get('lr_decay', None)

    if algo == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr= cfg_optim.get('lr', 1e-4), weight_decay= cfg_optim.get('weight_decay', 0))

    elif algo == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr= cfg_optim.get('lr', 1e-4), momentum= cfg_optim.get('momentum', 0), weight_decay= cfg_optim.get('weight_decay', 0))

    else:
        raise ValueError(f"Unsupported optimizer algorithm: {algo}")

    if lr_decay == None:
        scheduler = None

    elif lr_decay == 'exp':
        scheduler = ExponentialLR(optimizer= optimizer, gamma= cfg_optim.get('gamma', 0.9))
    
    elif lr_decay == 'step':
        scheduler = StepLR(optimizer= optimizer, step_size= 2, gamma= cfg_optim.get('gamma', 0.9))

    else:
        raise ValueError(f'Unsupported decay schedule: {lr_decay}')

    return optimizer, scheduler

def initialize_model(cfg_NN, device):

    model_type = cfg_NN.get('model_type')

    if model_type == 'MLP':
        model = MLP(cfg_MLP= cfg_NN).to(device)
    
    elif model_type == 'CNN1D_MLP':
        model= CNN1D_MLP(cfg_CNNMLP= cfg_NN).to(device)
    
    elif model_type == 'NIF_PartialPaper':
        model= NIF_PartialPaper(cfg_param_net= cfg_NN['cfg_param_net'], cfg_shape_net= cfg_NN['cfg_shape_net']).to(device)
    
    elif model_type == 'NIF_Pointwise':
        model= NIF_Pointwise(cfg_param_net= cfg_NN['cfg_param_net'], cfg_shape_net= cfg_NN['cfg_shape_net']).to(device)

    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return model