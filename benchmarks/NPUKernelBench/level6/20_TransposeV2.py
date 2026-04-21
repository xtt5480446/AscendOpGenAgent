import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """TransposeV2: permute tensor dimensions."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, perm: list) -> torch.Tensor:
        return torch.permute(x, perm)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "20_TransposeV2.json")
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
        x = torch.randn(x_info["shape"], dtype=dtype)
        perm = inputs[1]["value"]
        input_groups.append([x, perm])
    return input_groups


def get_init_inputs():
    return []
