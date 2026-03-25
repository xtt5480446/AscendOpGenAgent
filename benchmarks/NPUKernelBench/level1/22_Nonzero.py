# torch.nonzero(input, *, out=None, as_tuple=False)
# https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that returns indices of non-zero elements.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, as_tuple: bool = False):
        """
        Returns indices of non-zero elements in the tensor.

        Args:
            x (torch.Tensor): Input tensor.
            as_tuple (bool, optional): If True, returns a tuple of 1-D tensors, one for each dimension.

        Returns:
            torch.Tensor or tuple: Indices of non-zero elements.
        """
        return torch.nonzero(x, as_tuple=as_tuple)
