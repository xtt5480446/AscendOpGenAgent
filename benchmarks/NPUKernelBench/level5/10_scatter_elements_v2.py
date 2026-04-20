import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ScatterElementsV2: scatters updates into self along axis using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, index: torch.Tensor, updates: torch.Tensor,
                axis: int = 0, reduction: str = "none") -> torch.Tensor:
        """
        Args:
            self_tensor (torch.Tensor): target tensor.
            index (torch.Tensor): indices tensor.
            updates (torch.Tensor): values to scatter.
            axis (int): axis to scatter along.
            reduction (str): reduction mode.
        Returns:
            torch.Tensor: updated tensor.
        """
        out = self_tensor.clone()
        idx = index.long()
        if reduction == "add" or reduction == "sum":
            out = torch.scatter_add(out, axis, idx, updates)
        elif reduction == "multiply":
            # multiply reduction not directly supported, use loop
            for coord in torch.cartesian_prod(*[torch.arange(s) for s in idx.shape]):
                coord = tuple(int(c) for c in coord)
                i = idx[coord].item()
                target_coord = list(coord)
                target_coord[axis] = i
                out[tuple(target_coord)] *= updates[coord]
        else:
            out = torch.scatter(out, axis, idx, updates)
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 8]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "reduction", "type": "attr", "required": False, "dtype": "str", "value": "none"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 32]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [16, 32]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 32]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "reduction", "type": "attr", "required": False, "dtype": "str", "value": "add"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 16, 32]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [8, 16, 32]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 16, 32]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": 2},
            {"name": "reduction", "type": "attr", "required": False, "dtype": "str", "value": "none"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 4, 6, 8]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 4, 6, 8]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 4, 6, 8]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -1},
            {"name": "reduction", "type": "attr", "required": False, "dtype": "str", "value": "add"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [32, 64]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [32, 64]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [32, 64]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "reduction", "type": "attr", "required": False, "dtype": "str", "value": "none"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 128]},
            {"name": "index", "type": "tensor", "required": True, "dtype": "int32", "shape": [1, 128]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 128]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "reduction", "type": "attr", "required": False, "dtype": "str", "value": "add"},
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
        self_shape = None
        axis = 0
        for item in case["inputs"]:
            if item["name"] == "self":
                self_shape = item["shape"]
            if item["name"] == "axis":
                axis = item["value"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "index":
                    dim = axis if axis >= 0 else len(self_shape) + axis
                    max_idx = self_shape[dim]
                    t = torch.randint(0, max_idx, item["shape"]).to(dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                    if dtype == torch.int32:
                        t = (t * 10).clamp(-10000, 10000).to(torch.int32)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
