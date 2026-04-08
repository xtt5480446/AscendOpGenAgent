import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs transpose 2D convolution.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True) -> torch.Tensor:
        """
        Applies transpose 2D convolution to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True.

        Returns:
            torch.Tensor: Output tensor after performing nn.ConvTranspose2d.
        """
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        return conv(x)
