"""

Here, we define the loss functions:

    - abundance_wise_physical: allows initialization of abundance wise losses, and physicality loss (sums to 1, a percentage)

"""

import torch
import torch.nn as nn

class ClassWisePhysical(nn.Module):

    """
    Calculate the loss for predictions and target, 

    Parameters:
        - pred (torch.tensor) Prediction tensor
        - target (torch.tensor) Target tensor
        - cfg_loss (dict):
            - lambdas_ab (torch.tensor) A tensor of lambdas for each EM class prediction
            - lambda_sum (float) A lambda for the loss that ensures predictions sum up to one
            - lambda_bounds (float) A lambda for the deviation of predictions from [0,1] bounds

    Returns:
        - total_loss (float): Final loss meaned across the batch, for more detail, look inside
        - reconstruction_deviance (torch.tensor): Deviance of reconstruction for each EM class, meaned
        - sum_penalty (float): Deviance from sum up to 1 rule, meaned
        - bounds_penalty (float): Deviance from the bounds rule for each EM class, meaned
    """

    def __init__(self, cfg_loss):
        super().__init__()

        # Get lambdas
        self.lambdas_ab = cfg_loss.get('lambdas_ab', torch.tensor([0.5, 0.7, 1.0]))
        self.lambda_sum = cfg_loss.get('lambda_sum', 1)
        self.lambda_bounds = cfg_loss.get('lambda_bounds', 0.8)

    def forward(self, pred, target):

        # Reconstruction penalty
        reconstruction_deviance = (pred - target) ** 2
        reconstruction_loss = (reconstruction_deviance * self.lambdas_ab).sum(dim=1)

        # Sum to one penalty
        sum_loss = (pred.sum(dim=1) - 1) ** 2

        # Bounds penalty
        below_zero = torch.relu(-pred) # use relu(-x) to penalize vals only <0
        above_one = torch.relu(pred - 1) # use relu(x - 1) to penalize vals only >1
        bounds_loss = (below_zero + above_one).pow(2).sum(dim=1) # Sum pointwise, square, sum entirety

        total_loss = reconstruction_loss + self.lambda_sum * sum_loss + self.lambda_bounds * bounds_loss

        # Return the mean losses (it gets the mean loss at that batch, which becomes a good classifier)
        # Doing dim=0 on recons_dev allows us to preserve mean vals for each EM class prediction
        return total_loss.mean(), reconstruction_deviance.mean(dim=0), sum_loss.mean(), bounds_loss.mean()
