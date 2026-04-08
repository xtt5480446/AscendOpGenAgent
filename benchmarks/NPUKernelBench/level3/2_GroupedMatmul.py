import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs grouped matrix multiplication using torch.nn.functional.grouped_mm.
    Supports both 3D inputs (direct grouping) and 2D inputs with offset-based grouping.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor = None) -> torch.Tensor:
        """
        Applies grouped matrix multiplication between A and B.

        Args:
            A (torch.Tensor): Left operand tensor.
                - 3D shape: (num_groups, m, k) - groups are directly enumerated
                - 2D shape: (total_rows, k) - rows are grouped according to offsets
            B (torch.Tensor): Right operand tensor.
                - Shape: (num_groups, k, n) for common forward pass (out = input @ weight.T)
            offsets (torch.Tensor, optional): 1D tensor of monotonically increasing int32 offsets.
                Required when A is 2D, should be None when A is 3D.
                offsets[i] marks the end of group i in A's leading dimension.

        Returns:
            torch.Tensor: Concatenated results of each per-group GEMM operation.
        """
        return torch.nn.functional.grouped_mm(A, B, offs=offsets)
