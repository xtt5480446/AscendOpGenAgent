# torch.cat(tensors, dim=0, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that concatenates tensors along a dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, dim: int = 0) -> torch.Tensor:
        """
        Concatenates the given sequence of tensors in the given dimension.

        Args:
            tensors (list): List of tensors to concatenate. All tensors must have the same shape except in the concatenating dimension.
            dim (int, optional): The dimension over which the tensors are concatenated.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        return torch.cat(tensors, dim=dim)
