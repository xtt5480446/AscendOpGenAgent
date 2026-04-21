import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """UnfoldGrad: backward of unfold operation."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, input_sizes: list, dim: int, size: int, step: int) -> torch.Tensor:
        return torch.ops.aten.unfold_backward(grad_output, input_sizes, dim, size, step)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "21_UnfoldGrad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        grad_output = torch.randn(x_info["shape"], dtype=dtype)
        input_sizes = inputs[1]["value"]
        dim = inputs[2]["value"]
        size = inputs[3]["value"]
        step = inputs[4]["value"]
        input_groups.append([grad_output, input_sizes, dim, size, step])
    return input_groups


def get_init_inputs():
    return []
