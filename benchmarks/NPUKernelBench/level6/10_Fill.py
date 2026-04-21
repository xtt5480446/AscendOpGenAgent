import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """Fill: fill a tensor with a scalar value."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dims: list, value: float) -> torch.Tensor:
        return torch.full(dims, value)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "10_Fill.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dims = inputs[0]["value"]
        value = inputs[1]["value"]
        input_groups.append([dims, value])
    return input_groups


def get_init_inputs():
    return []
