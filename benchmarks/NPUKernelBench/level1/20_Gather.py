# torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.gather.html#torch-gather

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that gathers values along a dimension using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, sparse_grad: bool = False) -> torch.Tensor:
        """
        Gathers values along an axis specified by dim.

        Args:
            x (torch.Tensor): Input tensor (src).
            dim (int): The axis along which to index.
            index (torch.Tensor): The indices of elements to gather.
            sparse_grad (bool, optional): If True, gradient w.r.t. input will be a sparse tensor.

        Returns:
            torch.Tensor: Tensor with gathered values, same shape as index.
        """
        return torch.gather(x, dim, index, sparse_grad=sparse_grad)
