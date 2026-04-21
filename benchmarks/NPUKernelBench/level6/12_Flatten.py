import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """Flatten: flatten a tensor to 2D starting from a given axis."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, axis: int = 1) -> torch.Tensor:
        return torch.flatten(x, start_dim=axis)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "12_Flatten.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        axis = inputs[1]["value"]
        input_groups.append([x, axis])
    return input_groups


def get_init_inputs():
    return []
