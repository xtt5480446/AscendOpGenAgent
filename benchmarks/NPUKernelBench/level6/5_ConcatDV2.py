import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """ConcatDV2: concatenate tensors along a dimension."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, concat_dim: int = 0) -> torch.Tensor:
        return torch.cat(tensors, dim=concat_dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_ConcatDV2.json")
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
        tensors_info = inputs[0]
        dtype = dtype_map[tensors_info["dtype"]]
        shapes = tensors_info["shapes"]
        tensors = [torch.randn(shape, dtype=dtype) for shape in shapes]
        concat_dim = inputs[1]["value"]
        input_groups.append([tensors, concat_dim])
    return input_groups


def get_init_inputs():
    return []
