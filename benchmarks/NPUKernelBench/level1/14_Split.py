# torch.split(tensor, split_size_or_sections, dim=0)
# https://docs.pytorch.org/docs/stable/generated/torch.split.html#torch.split

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that splits a tensor into chunks.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, split_size_or_sections, dim: int = 0):
        """
        Splits the tensor into chunks.

        Args:
            x (torch.Tensor): Input tensor to split.
            split_size_or_sections (int or list): If int, size of each chunk. If list, sizes of each chunk.
            dim (int, optional): Dimension along which to split the tensor.

        Returns:
            tuple: Tuple of tensors resulting from the split.
        """
        return torch.split(x, split_size_or_sections, dim=dim)
