# torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)
# https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that repeats elements of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats, dim: int = None, output_size: int = None) -> torch.Tensor:
        """
        Repeats elements of a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (int or torch.Tensor): Number of repetitions for each element.
            dim (int, optional): The dimension along which to repeat values.
            output_size (int, optional): Total output size for the repeated dimension.

        Returns:
            torch.Tensor: Tensor with repeated elements.
        """
        if output_size is not None:
            return torch.repeat_interleave(x, repeats, dim=dim, output_size=output_size)
        return torch.repeat_interleave(x, repeats, dim=dim)
