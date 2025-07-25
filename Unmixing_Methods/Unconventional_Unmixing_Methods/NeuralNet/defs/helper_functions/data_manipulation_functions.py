from torch.utils.data import DataLoader
import random
import numpy as np
import torch

"""

Here, we define a way to load our data for:

    - K-Fold-Crossval_Data: Splits and defines functions to get folds of a data

"""

class K_Fold_Crossval_Data:

    """
    Prepares data for K-Fold-Crossval Training

    Parameters
        - dataset (torch.tensor): The entire dataset to be used as a tensor, must include true indices and true spectral names since we are shuffling

        - cfg_train (dict): A dictionary that contains:
            - fold (int): How many folds we want

        - cfg_dataset (dict): A dictionary to process dataset that contains:
            - abundance_idx_range (list(int)): A list that specifies start-end(included) indices of abundances
            - value_idx_range (list(int)): A list that specifies start-end(included) indices of spectral values
            - identity_idx_range (list(int)): A list that specifies start-end(included) indice(s) of name, and original index
            - property_idx_range (list(int)): A list that specifies start-end(included) indice(s) of additional properties such as RWC, VZA, etc.
            - seed (int): For replicibility of shuffle, def=69

    Returns
        - Training and validation data at kth fold when get_data_at_fold is used
    
    """

    def __init__(self, dataset: torch.tensor, cfg_train: dict, cfg_dataset: dict):
        super().__init__()

        # Retreive the seed to shuffle
        self.seed = cfg_dataset.get('seed', 69)
        np.random.seed(self.seed)
        self.full_dataset = dataset.index_select(0, torch.randperm(dataset.size(0)), generator= torch.manual_seed(self.seed))

        # Get the folds to seperate the dataset into chunks
        self.fold = cfg_train['fold']
        assert ((self.full_dataset.shape[0])%self.fold == 0), "The dataset's rows must be divisible by the fold, to ensure uniformity"

        self.abundance_idx_range = cfg_dataset['abundance_idx_range']
        self.abundance_dataset = self.full_dataset[:, self.abundance_idx_range[0]: self.abundance_idx_range[1] + 1] # Pull out the feature
        self.abundance_dataset = torch.tensor_split(self.abundance_dataset, self.fold, dim=0) # Generate folds

        self.value_idx_range = cfg_dataset['value_idx_range']
        self.value_dataset = self.full_dataset[:, self.value_idx_range[0]: self.value_idx_range[1] + 1] # Pull out the feature
        self.value_dataset = torch.tensor_split(self.value_dataset, self.fold, dim=0) # Generate folds

        self.identity_idx_range = cfg_dataset['identity_idx_range']
        self.identity_dataset = self.full_dataset[:, self.identity_idx_range[0]: self.identity_idx_range[1] + 1] # Pull out the feature
        self.identity_dataset = torch.tensor_split(self.identity_dataset, self.fold, dim=0) # Generate folds

        # See if we want to return the properties as well
        self.property_idx_range = cfg_dataset.get('property_idx_range', False)

        if self.property_idx_range is not False: 
            self.property_dataset = self.full_dataset[:, self.property_idx_range[0]: self.property_idx_range[1] + 1] # Pull out the feature
            self.property_dataset = torch.tensor_split(self.property_dataset, self.fold, dim=0) # Generate folds

        del self.full_dataset # Delete the giant, useless dataset, to preserve memory


    def get_data_at_fold(self, fold: int):

        """
        Returns the data at that kth fold.

        Parameters
            - fold (int): At which fold do we want to return the data

        Returns
            - train_abundances (torch.tensor)
            - train_values (torch.tensor)
            - train_identities (torch.tensor)
            - train_properties (torch.tensor), if enabled

            - val_abundances (torch.tensor)
            - val_values (torch.tensor)
            - val_identities (torch.tensor)
            - val_properties (torch.tensor), if enabled
        """

        assert 0 <= fold <= self.fold, f"The specific fold must be between (and including): 0 and {self.fold -1}"

        val_abundaces = self.abundance_dataset[fold]
        val_values = self.value_dataset[fold]
        val_identities = self.identity_dataset[fold]
        
        train_abundances = torch.cat([data for i, data in enumerate(self.abundance_dataset) if i != fold], dim = 0)
        train_values = torch.cat([data for i, data in enumerate(self.value_dataset) if i != fold], dim = 0)
        train_identities = torch.cat([data for i, data in enumerate(self.identity_dataset) if i != fold], dim = 0)

        if self.property_idx_range is not None:
            val_properties = self.property_dataset[fold]
            train_properties = torch.cat([data for i, data in enumerate(self.property_dataset) if i != fold], dim = 0)

            return train_abundances, train_values, train_identities, train_properties, val_abundaces, val_values, val_identities, val_properties

        else:

            return train_abundances, train_values, train_identities, val_abundaces, val_values, val_identities