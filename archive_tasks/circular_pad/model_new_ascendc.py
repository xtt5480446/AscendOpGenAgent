import os
import sys

import torch

_kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel", "build")
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

import _circular_pad_ext


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, padding):
        return _circular_pad_ext.run_circular_pad(x, padding)


def get_input_groups():
    from model import get_input_groups as _get_input_groups
    return _get_input_groups()


def get_init_inputs():
    return []
