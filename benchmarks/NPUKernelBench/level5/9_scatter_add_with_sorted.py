import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ScatterAddWithSorted: adds value to self using sorted_index and pos.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, value: torch.Tensor,
                sorted_index: torch.Tensor, pos: torch.Tensor, reduction: str) -> torch.Tensor:
        """
        Args:
            self_tensor (torch.Tensor): target tensor.
            value (torch.Tensor): values to scatter.
            sorted_index (torch.Tensor): sorted indices.
            pos (torch.Tensor): position tensor.
            reduction (str): reduction mode.
        Returns:
            torch.Tensor: updated tensor.
        """
        out = self_tensor.clone()
        idx = sorted_index.long()
        # Use scatter_add as reference
        if reduction == "add" or reduction == "sum":
            out = torch.scatter_add(out, 0, idx.unsqueeze(1).expand_as(value), value)
        elif reduction == "multiply":
            # scatter multiply is not standard in torch, use loop
            for i in range(idx.shape[0]):
                out[idx[i]] *= value[i]
        else:
            for i in range(idx.shape[0]):
                out[idx[i]] = value[i]
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [65, 4096]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float32", "shape": [63, 4096]},
            {"name": "sorted_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [63]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [63]},
            {"name": "reduction", "type": "attr", "required": True, "dtype": "str", "value": "add"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [128, 256]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float32", "shape": [64, 256]},
            {"name": "sorted_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [64]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [64]},
            {"name": "reduction", "type": "attr", "required": True, "dtype": "str", "value": "add"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float16", "shape": [32, 1024]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 1024]},
            {"name": "sorted_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "reduction", "type": "attr", "required": True, "dtype": "str", "value": "add"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16, 512]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 512]},
            {"name": "sorted_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "reduction", "type": "attr", "required": True, "dtype": "str", "value": "add"},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 4, 128]},
            {"name": "value", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 4, 128]},
            {"name": "sorted_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "pos", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "reduction", "type": "attr", "required": True, "dtype": "str", "value": "add"},
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
        for item in case["inputs"]:
            if item["name"] == "self":
                self_shape = item["shape"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "sorted_index":
                    n = item["shape"][0]
                    max_idx = self_shape[0]
                    t = torch.sort(torch.randint(0, max_idx, (n,)))[0].to(dtype)
                elif item["name"] == "pos":
                    n = item["shape"][0]
                    t = torch.arange(n).to(dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
