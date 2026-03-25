# torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs interpolation (resizing) of tensors.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, size=None, scale_factor=None,
                mode: str = 'nearest', align_corners=None,
                recompute_scale_factor=None, antialias: bool = False) -> torch.Tensor:
        """
        Interpolates (resizes) the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, ...) where ... represents spatial dimensions.
            size (optional): Output spatial size.
            scale_factor (optional): Multiplier for spatial size.
            mode (str, optional): Algorithm used for interpolation: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'.
            align_corners (optional): How to align corners when resizing.
            recompute_scale_factor (optional): Recompute scale_factor for backward compatibility.
            antialias (bool, optional): Apply antialiasing.

        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return torch.nn.functional.interpolate(
            x, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )
