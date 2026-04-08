import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs batch matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Applies batch matrix multiplication between A and B.

        Args:
            A (torch.Tensor): Input tensor of shape (batch, m, n).
            B (torch.Tensor): Input tensor of shape (batch, n, p).

        Returns:
            torch.Tensor: Output tensor of shape (batch, m, p) after performing torch.bmm.
        """
        return torch.bmm(A, B)
