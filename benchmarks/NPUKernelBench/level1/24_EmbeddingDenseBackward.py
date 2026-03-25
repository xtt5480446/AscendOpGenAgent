# torch.ops.aten.embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq) -> Tensor

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs the backward pass for embedding with dense gradients.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, indices: torch.Tensor,
                num_weights: int, padding_idx: int = -1, scale_grad_by_freq: bool = False) -> torch.Tensor:
        """
        Computes the gradient for embedding layer with dense backward.

        Args:
            grad_output (torch.Tensor): Gradient of the output.
            indices (torch.Tensor): The indices tensor from forward pass.
            num_weights (int): Number of rows in the embedding weight matrix.
            padding_idx (int, optional): Index of padding token to zero out gradient.
            scale_grad_by_freq (bool, optional): Whether to scale gradients by frequency.

        Returns:
            torch.Tensor: Gradient tensor for embedding weights.
        """
        return torch.ops.aten.embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)
