# Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that scatters values from src into a tensor at specified indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, reduce: str = None) -> torch.Tensor:
        """
        Scatters values from src into x at the indices specified in index.

        Args:
            x (torch.Tensor): Input tensor (self, will be cloned to avoid in-place modification).
            dim (int): The axis along which to index.
            index (torch.Tensor): The indices of elements to scatter.
            src (torch.Tensor): The source elements to scatter.
            reduce (str, optional): Reduction operation ('add', 'multiply').

        Returns:
            torch.Tensor: Tensor with scattered values.
        """
        if reduce is not None:
            x.scatter_(dim, index, src, reduce=reduce)
        else:
            x.scatter_(dim, index, src)
        return x
