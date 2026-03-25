# torch.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-05)
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.group_norm.html#torch.nn.functional.group_norm

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that applies Group Normalization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, num_groups: int, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Group Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, *].
            num_groups (int): Number of groups to separate channels into.
            weight (torch.Tensor, optional): Weight tensor of shape [C].
            bias (torch.Tensor, optional): Bias tensor of shape [C].

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.group_norm(x, num_groups, weight=weight, bias=bias)
