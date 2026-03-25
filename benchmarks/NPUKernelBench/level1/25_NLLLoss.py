# torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the negative log likelihood loss.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor = None, ignore_index: int = -100,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the negative log likelihood loss.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C) or (N, C, d1, d2, ...).
            target (torch.Tensor): Target tensor of shape (N,) or (N, d1, d2, ...).
            weight (torch.Tensor, optional): Manual rescaling weight given to each class.
            ignore_index (int, optional): Target value that is ignored and does not contribute to gradient.
            reduction (str, optional): Reduction method ('none', 'mean', 'sum').

        Returns:
            torch.Tensor: NLL loss value.
        """
        return torch.nn.functional.nll_loss(input, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
