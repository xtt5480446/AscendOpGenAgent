# torch.index_select(input, dim, index, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.index_select.html#torch.index_select

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that selects elements from a tensor along a dimension using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        """
        Selects elements from input tensor along the specified dimension using index.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension in which to index.
            index (torch.Tensor): The 1-D tensor containing the indices to index.

        Returns:
            torch.Tensor: Tensor with selected elements.
        """
        return torch.index_select(x, dim, index)
