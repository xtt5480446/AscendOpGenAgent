import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, padding: list):
        return F.pad(x, tuple(padding), mode='circular')


INPUT_CASES = [
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 1, 3, 3]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 1, 300, 300]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [100, 100, 100, 100]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 1, 3, 3]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 1, 300, 300]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [100, 100, 100, 100, 1, 1]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 3, 5, 5]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [2, 1, 1, 2]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 3, 5, 5]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 2, 1, 1, 1]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "int8", "shape": [1, 2, 4, 4]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [1, 1, 6, 6]},
            {"name": "padding", "type": "attr", "required": True, "dtype": "list", "value": [3, 3, 3, 3]},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int8": torch.int8,
    "int32": torch.int32,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                t = torch.randn(item["shape"], dtype=torch.float32 if dtype in (torch.int8, torch.int32) else dtype)
                if dtype == torch.int8:
                    t = (t * 10).clamp(-127, 127).to(torch.int8)
                elif dtype == torch.int32:
                    t = (t * 10).clamp(-10000, 10000).to(torch.int32)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
