import torch
import torch.nn as nn


class Model(nn.Module):
    """
    IndexPutWithSort: scatter updates into self according to linear_index and pos_idx.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, linear_index: torch.Tensor,
                pos_idx: torch.Tensor, values: torch.Tensor,
                slice_size: int, accumulate: bool) -> torch.Tensor:
        """
        Args:
            self_tensor (torch.Tensor): target tensor to update.
            linear_index (torch.Tensor): [indices], points into self.
            pos_idx (torch.Tensor): [indices], points into values.
            values (torch.Tensor): source values.
            slice_size (int): slice size per index.
            accumulate (bool): whether to accumulate.
        Returns:
            torch.Tensor: updated self tensor.
        """
        out = self_tensor.clone()
        indices = linear_index.long()
        pos = pos_idx.long()
        for i in range(indices.shape[0]):
            idx = indices[i].item()
            p = pos[i].item()
            if accumulate:
                out[idx] += values[p]
            else:
                out[idx] = values[p]
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 1536]},
            {"name": "linear_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "pos_idx", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 1536]},
            {"name": "slice_size", "type": "attr", "required": True, "dtype": "int", "value": 1536},
            {"name": "accumulate", "type": "attr", "required": True, "dtype": "bool", "value": False},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 1536]},
            {"name": "linear_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "pos_idx", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 1536]},
            {"name": "slice_size", "type": "attr", "required": True, "dtype": "int", "value": 1536},
            {"name": "accumulate", "type": "attr", "required": True, "dtype": "bool", "value": True},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 512]},
            {"name": "linear_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "pos_idx", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "values", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 512]},
            {"name": "slice_size", "type": "attr", "required": True, "dtype": "int", "value": 512},
            {"name": "accumulate", "type": "attr", "required": True, "dtype": "bool", "value": False},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 256]},
            {"name": "linear_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "pos_idx", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "values", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 256]},
            {"name": "slice_size", "type": "attr", "required": True, "dtype": "int", "value": 256},
            {"name": "accumulate", "type": "attr", "required": True, "dtype": "bool", "value": True},
        ]
    },
    {
        "inputs": [
            {"name": "self", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 128]},
            {"name": "linear_index", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "pos_idx", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 128]},
            {"name": "slice_size", "type": "attr", "required": True, "dtype": "int", "value": 128},
            {"name": "accumulate", "type": "attr", "required": True, "dtype": "bool", "value": False},
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
                if item["name"] == "linear_index":
                    n = item["shape"][0]
                    t = torch.randint(0, self_shape[0], (n,)).to(dtype)
                elif item["name"] == "pos_idx":
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
