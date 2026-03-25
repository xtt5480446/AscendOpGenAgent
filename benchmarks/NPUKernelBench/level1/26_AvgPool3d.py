# torch.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool3d.html#torch.nn.functional.avg_pool3d

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 3D average pooling.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                ceil_mode: bool = False, count_include_pad: bool = True,
                divisor_override=None) -> torch.Tensor:
        """
        Applies 3D average pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            count_include_pad (bool, optional): Include zero-padding in averaging.
            divisor_override (optional): If specified, will be used as divisor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return torch.nn.functional.avg_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )
