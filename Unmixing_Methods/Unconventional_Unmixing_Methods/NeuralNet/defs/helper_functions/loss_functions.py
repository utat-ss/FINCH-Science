"""

Here, we define the loss functions:

    - abundance_wise_physical: allows initialization of abundance wise losses, and physicality loss (sums to 1, a percentage)

"""

import torch

def abundance_wise_physical(pred: torch.tensor, target: torch.tensor, lambdas_ab: torch.tensor, lambda_sum: float, lambda_bounds: float) -> torch.tensor:

    """
    Calculate the loss for predictions and target, 

    Parameters:
        - pred (torch.tensor) Prediction tensor
        - target (torch.tensor) Target tensor
        - lambdas_ab (torch.tensor) A tensor of lambdas for each EM class prediction
        - lambda_sum (float) A lambda for the loss that ensures predictions sum up to one
        - lambda_bounds (float) A lambda for the deviation of predictions from [0,1] bounds

    Returns:
        - total_loss (float): Final loss meaned across the batch, for more detail, look inside
        - reconstruction_deviance (torch.tensor): Deviance of reconstruction for each EM class
        - sum_penalty (float): Deviance from sum up to 1 rule
        - bounds_penalty (float): Deviance from the bounds rule for each EM class
    """

    # Reconstruction penalty
    reconstruction_deviance = (pred - target) ** 2
    reconstruction_penalty = (reconstruction_deviance * lambdas_ab).sum(dim=1)

    # Sum to one penalty
    sum_penalty = (pred.sum(dim=1) - 1) ** 2

    # Bounds penalty
    below_zero = torch.relu(-pred) # use relu(-x) to penalize vals only <0
    above_one = torch.relu(pred - 1) # use relu(x - 1) to penalize vals only >1
    bounds_penalty = (below_zero + above_one).pow(2).sum(dim=1) # Sum pointwise, square, sum entirety

    total_loss = reconstruction_penalty + lambda_sum * sum_penalty + lambda_bounds * bounds_penalty

    return total_loss.mean(), reconstruction_deviance, sum_penalty, bounds_penalty