import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs absolute value operation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies absolute value to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with absolute values, same shape as input.
        """
        return torch.abs(x)
