import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ScatterSub: subtracts updates from var at positions specified by indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            var (torch.Tensor): tensor to update.
            indices (torch.Tensor): indices tensor.
            updates (torch.Tensor): values to subtract.
        Returns:
            torch.Tensor: updated var.
        """
        out = var.clone()
        idx = indices.long()
        for i in range(idx.shape[0]):
            out[idx[i]] -= updates[i]
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8]},
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8]},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 256]},
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 256]},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float16", "shape": [32, 1024]},
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 1024]},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "int32", "shape": [8, 128]},
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 128]},
        ]
    },
    {
        "inputs": [
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [64, 64, 64]},
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [32]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 64, 64]},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int32": torch.int32,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        var_shape = None
        for item in case["inputs"]:
            if item["name"] == "var":
                var_shape = item["shape"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "indices":
                    n = item["shape"][0]
                    max_idx = var_shape[0]
                    t = torch.randint(0, max_idx, (n,)).to(dtype)
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
