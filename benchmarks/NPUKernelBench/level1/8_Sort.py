# torch.sort(input, dim=-1, descending=False, *, stable=False, out=None)
# https://docs.pytorch.org/docs/stable/generated/torch.sort.html#torch.sort

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that sorts a tensor along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
        """
        Sorts the elements of the input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int): The dimension to sort along.
            descending (bool): Controls the sorting order (ascending or descending).

        Returns:
            torch.Tensor: Sorted tensor with same shape as input.
        """
        values, indices = torch.sort(x, dim=dim, descending=descending)
        return values
