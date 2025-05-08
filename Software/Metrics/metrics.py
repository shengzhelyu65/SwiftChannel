import numpy as np
import torch
import torch.nn as nn

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, Hest, H):
        # Ensure H and Hest have the same shape
        assert H.shape == Hest.shape, "H and Hest must have the same shape"

        # Determine the number of dimensions
        num_dims = len(H.shape)
        
        # Calculate the numerator: ||H - Hest||^2_2
        # Sum over all dimensions except the first one (batch dimension)
        numerator_dims = tuple(range(1, num_dims))
        numerator = torch.sum((H - Hest) ** 2, dim=numerator_dims)
        
        # Calculate the denominator: ||H||^2_2
        denominator = torch.sum(H ** 2, dim=numerator_dims)

        # Compute the NMSE for each sample in the batch
        nmse = numerator / denominator
        
        # Return the mean NMSE over the batch
        loss = torch.mean(nmse)
        
        return loss
    
