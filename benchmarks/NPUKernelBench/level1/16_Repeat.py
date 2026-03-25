# Tensor.repeat(*repeats) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch.Tensor.repeat

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that repeats a tensor along specified dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats: tuple) -> torch.Tensor:
        """
        Repeats the tensor along each dimension.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (tuple): Number of repeats for each dimension.

        Returns:
            torch.Tensor: Repeated tensor.
        """
        return x.repeat(*repeats)
