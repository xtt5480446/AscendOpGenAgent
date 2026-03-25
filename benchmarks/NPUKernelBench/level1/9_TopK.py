# torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
# https://docs.pytorch.org/docs/stable/generated/torch.topk.html#torch.topk

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that returns the k largest elements of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, k: int, dim: int = -1, largest: bool = True) -> torch.Tensor:
        """
        Returns the k largest/smallest elements of the input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            k (int): The number of elements to return.
            dim (int): The dimension to sort along.
            largest (bool): If True, return largest k elements; otherwise smallest.

        Returns:
            torch.Tensor: Tensor of top k values with shape [..., k, ...].
        """
        values, indices = torch.topk(x, k, dim=dim, largest=largest)
        return values
