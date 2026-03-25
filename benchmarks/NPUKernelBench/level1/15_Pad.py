# torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs padding on a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, pad: tuple, mode: str = 'constant', value: float = None) -> torch.Tensor:
        """
        Pads tensor with specified padding mode.

        Args:
            x (torch.Tensor): Input tensor.
            pad (tuple): Padding sizes in the form (pad_left, pad_right, pad_top, pad_bottom, ...).
            mode (str, optional): Padding mode: 'constant', 'reflect', 'replicate', 'circular'.
            value (float, optional): Fill value for 'constant' padding.

        Returns:
            torch.Tensor: Padded tensor.
        """
        return torch.nn.functional.pad(x, pad, mode=mode, value=value)
