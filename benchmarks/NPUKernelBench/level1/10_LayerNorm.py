# torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html#torch.nn.functional.layer_norm

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that applies Layer Normalization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, normalized_shape: list, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Layer Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [*, normalized_shape[0], ...].
            normalized_shape (list): Shape over which to normalize.
            weight (torch.Tensor, optional): Weight tensor of shape normalized_shape.
            bias (torch.Tensor, optional): Bias tensor of shape normalized_shape.

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias)
