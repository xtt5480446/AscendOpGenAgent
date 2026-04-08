import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication with transposed A.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Applies matrix multiplication between transposed A and B.

        Args:
            A (torch.Tensor): Input tensor of shape (*, m, n).
            B (torch.Tensor): Input tensor of shape (*, n, k).

        Returns:
            torch.Tensor: Output tensor of shape (*, n, k) after performing torch.matmul(A.T, B).
        """
        return torch.matmul(A.T, B)
