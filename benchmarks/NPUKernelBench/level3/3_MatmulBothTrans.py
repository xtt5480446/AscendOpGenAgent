import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication with both A and B transposed.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Applies matrix multiplication between transposed A and transposed B.

        Args:
            A (torch.Tensor): Input tensor of shape (*, m, n).
            B (torch.Tensor): Input tensor of shape (*, k, p).

        Returns:
            torch.Tensor: Output tensor after performing torch.matmul(A.T, B.T).
        """
        return torch.matmul(A.T, B.T)
