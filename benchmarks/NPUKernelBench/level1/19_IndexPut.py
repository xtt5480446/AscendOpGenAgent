# Tensor.index_put_(indices, values, accumulate=False) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html#torch.Tensor.index_put_

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that puts values into a tensor at specified indices (1D case).
    For a 1D tensor x, uses a single index tensor to put values.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, values: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        """
        Puts values into the tensor at the specified indices.

        Args:
            x (torch.Tensor): Input tensor.
            index (torch.Tensor): 1-D index tensor for the first dimension.
            values (torch.Tensor): Values to put at the specified indices.
            accumulate (bool, optional): Whether to accumulate values at the indices.

        Returns:
            torch.Tensor: Tensor with values put at specified indices.
        """
        x.index_put_((index,), values, accumulate=accumulate)
        return x
