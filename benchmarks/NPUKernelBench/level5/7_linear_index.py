import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LinearIndex: maps multi-dimensional indices to a single linear index.
    Equivalent to np.ravel_multi_index.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, indices: torch.Tensor, var: torch.Tensor, axis: int = -1, combine: bool = False) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): multi-dimensional indices.
            var (torch.Tensor): reference tensor whose shape defines the dimensions.
            axis (int): axis along which to compute linear index.
            combine (bool): whether to combine output into 1D.
        Returns:
            torch.Tensor: linear indices.
        """
        shape = list(var.shape)
        idx_np = indices.cpu().numpy()
        out_shape = list(indices.shape)
        out = torch.zeros(out_shape, dtype=torch.int32)
        strides = [1]
        for i in range(len(shape) - 2, -1, -1):
            strides.insert(0, strides[0] * shape[i + 1])
        for coord in torch.cartesian_prod(*[torch.arange(s) for s in out_shape]):
            coord = [int(c) for c in coord]
            idx_vals = [int(idx_np[tuple(coord + [d])]) for d in range(indices.shape[-1])]
            linear = sum(v * s for v, s in zip(idx_vals, strides[:len(idx_vals)]))
            out[tuple(coord)] = linear
        if combine:
            out = out.reshape(-1)
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [100, 3]},
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [10, 10, 10]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -1},
            {"name": "combine", "type": "attr", "required": False, "dtype": "bool", "value": False},
        ]
    },
    {
        "inputs": [
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [1000, 2]},
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [100, 100]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -1},
            {"name": "combine", "type": "attr", "required": False, "dtype": "bool", "value": False},
        ]
    },
    {
        "inputs": [
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int64", "shape": [500, 4]},
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 16, 16, 16]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -1},
            {"name": "combine", "type": "attr", "required": False, "dtype": "bool", "value": True},
        ]
    },
    {
        "inputs": [
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [50, 1]},
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [100]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -1},
            {"name": "combine", "type": "attr", "required": False, "dtype": "bool", "value": False},
        ]
    },
    {
        "inputs": [
            {"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [10000, 2]},
            {"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [256, 256]},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -1},
            {"name": "combine", "type": "attr", "required": False, "dtype": "bool", "value": True},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
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
                    ndim = item["shape"][-1]
                    max_vals = [var_shape[i] for i in range(ndim)]
                    t = torch.stack([torch.randint(0, max_vals[d], (item["shape"][0],)) for d in range(ndim)], dim=-1).to(dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
