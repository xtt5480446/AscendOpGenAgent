# torch.histc(input, bins=100, min=0, max=0, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.histc.html#torch.histc

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the histogram of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, bins: int = 100, min_val: float = 0.0, max_val: float = 0.0) -> torch.Tensor:
        """
        Computes the histogram of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            bins (int): Number of histogram bins.
            min_val (float): Lower end of the range (inclusive).
            max_val (float): Upper end of the range (inclusive). If 0, uses tensor max.

        Returns:
            torch.Tensor: Histogram tensor of shape (bins,).
        """
        return torch.histc(x, bins=bins, min=min_val, max=max_val)
