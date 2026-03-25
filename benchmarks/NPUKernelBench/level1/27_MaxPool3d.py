# torch.nn.functional.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool3d.html#torch.nn.functional.max_pool3d

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 3D max pooling.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                dilation: int = 1, ceil_mode: bool = False,
                return_indices: bool = False):
        """
        Applies 3D max pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            dilation (int, optional): Spacing between kernel elements.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            return_indices (bool, optional): Return indices of max values.

        Returns:
            torch.Tensor or tuple: Pooled tensor (and indices if return_indices=True).
        """
        return torch.nn.functional.max_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            dilation=dilation, ceil_mode=ceil_mode,
            return_indices=return_indices
        )
