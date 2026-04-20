import torch
import torch.nn as nn


class Model(nn.Module):
    """
    GatherElementsV2: gathers values along an axis specified by index.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
            index (torch.Tensor): indices tensor, same dtype as x on gather dim.
            dim (int): axis to gather along.
        Returns:
            torch.Tensor: gathered tensor.
        """
        return torch.gather(x, dim, index)


INPUT_CASES = [
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 8]},
            {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 32]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [16, 32]},
            {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": 1},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 16, 32]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [8, 16, 32]},
            {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": 2},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 4, 6, 8]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 4, 6, 8]},
            {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": -1},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [32, 64]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [32, 64]},
            {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 128]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [1, 128]},
            {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": 1},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        dim = 0
        x_shape = None
        for item in case["inputs"]:
            if item["name"] == "x":
                x_shape = item["shape"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "index":
                    dim_val = next((i["value"] for i in case["inputs"] if i["name"] == "dim"), 0)
                    dim = dim_val if dim_val >= 0 else len(x_shape) + dim_val
                    max_idx = x_shape[dim]
                    t = torch.randint(0, max_idx, item["shape"]).to(dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
