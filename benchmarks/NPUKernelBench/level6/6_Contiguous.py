import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """Contiguous: make a non-contiguous tensor contiguous."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_Contiguous.json")
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
        if x_info.get("non_contiguous", False):
            x = x.transpose(0, -1)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
