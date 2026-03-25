import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs cumulative sum along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Applies cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int): The dimension to do the operation over.

        Returns:
            torch.Tensor: Output tensor with cumulative sum, same shape as input.
        """
        return torch.cumsum(x, dim=dim)
