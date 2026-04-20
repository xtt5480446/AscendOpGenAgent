import torch
import torch.nn as nn


class Model(nn.Module):
    """
    InplaceIndexAddWithSorted: adds alpha * value to var along axis using sorted_indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_indices: torch.Tensor,
                pos: torch.Tensor, alpha: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Args:
            var (torch.Tensor): tensor to be updated.
            value (torch.Tensor): values to add.
            sorted_indices (torch.Tensor): sorted indices.
            pos (torch.Tensor): positions in original index sequence.
            alpha (torch.Tensor): scalar scaling factor.
            axis (int): axis along which to add.
        Returns:
            torch.Tensor: updated var.
        """
        out = var.clone()
        alpha_val = alpha.item() if alpha.numel() == 1 else 1.0
        idx = sorted_indices.long()
        out = torch.index_add(out, axis, idx, value * alpha_val)
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [65, 4096]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float32", "shape": [63, 4096]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [63]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [63]},
            {"name": "alpha", "type": "tensor", "required": False, "dtype": "float32", "shape": [1]},
            {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [128, 256]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float32", "shape": [64, 256]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [64]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [64]},
            {"name": "alpha", "type": "tensor", "required": False, "dtype": "float32", "shape": [1]},
            {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float16", "shape": [32, 1024]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 1024]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "alpha", "type": "tensor", "required": False, "dtype": "float32", "shape": [1]},
            {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "int32", "shape": [16, 128]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "int32", "shape": [8, 128]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "alpha", "type": "tensor", "required": False, "dtype": "int32", "shape": [1]},
            {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 4, 256]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 4, 256]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "alpha", "type": "tensor", "required": False, "dtype": "float32", "shape": [1]},
            {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int16": torch.int16,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        var_shape = None
        axis = 0
        for item in case["inputs"]:
            if item["name"] == "var":
                var_shape = item["shape"]
            if item["name"] == "axis":
                axis = item["value"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "sorted_indices":
                    n = item["shape"][0]
                    max_idx = var_shape[axis]
                    t = torch.sort(torch.randint(0, max_idx, (n,)))[0].to(dtype)
                elif item["name"] == "pos":
                    n = item["shape"][0]
                    t = torch.arange(n).to(dtype)
                elif item["name"] == "alpha":
                    t = torch.tensor([1.5], dtype=dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
