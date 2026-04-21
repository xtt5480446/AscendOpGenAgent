import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """PadV3GradReplication: backward of replication padding 3d."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        grad_input = torch.nn.functional.pad(grad_output, pad, mode='replicate')
        return grad_input


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "15_PadV3GradReplication.json")
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
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
