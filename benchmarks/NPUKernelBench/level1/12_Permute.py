# torch.permute(input, dims) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.permute.html#torch.permute

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a permutation of tensor dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dims: tuple) -> torch.Tensor:
        """
        Permutes the dimensions of the input tensor.

        Args:
            x (torch.Tensor): Input tensor with at least the number of dimensions in dims.
            dims (tuple): The desired ordering of dimensions.

        Returns:
            torch.Tensor: Tensor with permuted dimensions.
        """
        return torch.permute(x, dims).contiguous()
