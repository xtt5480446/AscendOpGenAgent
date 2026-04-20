import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ScatterList: scatters updates into var list according to indices and mask.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var_list: list, indice: torch.Tensor, updates: torch.Tensor,
                mask: torch.Tensor = None, reduce: str = "update", axis: int = -2) -> list:
        """
        Args:
            var_list (list): list of tensors to update.
            indice (torch.Tensor): indices tensor.
            updates (torch.Tensor): values to scatter.
            mask (torch.Tensor, optional): mask tensor.
            reduce (str): reduction mode.
            axis (int): axis to scatter along.
        Returns:
            list: updated list of tensors.
        """
        out_list = [v.clone() for v in var_list]
        idx = indice.long()
        # Simplified reference: scatter updates into each var in list
        for i, var in enumerate(out_list):
            if mask is not None and not mask[i].item():
                continue
            if reduce == "update":
                out_list[i] = torch.index_copy(var, axis, idx, updates)
            elif reduce == "add":
                out_list[i] = torch.index_add(var, axis, idx, updates)
            elif reduce == "multiply":
                # not directly supported
                pass
        return out_list


INPUT_CASES = [
    {
        "inputs": [
            {"name": "var_list", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 16]},
            {"name": "indice", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 16]},
            {"name": "mask", "type": "tensor", "required": False, "dtype": "uint8", "shape": [4]},
            {"name": "reduce", "type": "attr", "required": False, "dtype": "str", "value": "update"},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -2},
        ]
    },
    {
        "inputs": [
            {"name": "var_list", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 16, 32]},
            {"name": "indice", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 16, 32]},
            {"name": "mask", "type": "tensor", "required": False, "dtype": "uint8", "shape": [8]},
            {"name": "reduce", "type": "attr", "required": False, "dtype": "str", "value": "add"},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -2},
        ]
    },
    {
        "inputs": [
            {"name": "var_list", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 4, 8, 16]},
            {"name": "indice", "type": "tensor", "required": True, "dtype": "int32", "shape": [2]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 4, 8, 16]},
            {"name": "mask", "type": "tensor", "required": False, "dtype": "uint8", "shape": [2]},
            {"name": "reduce", "type": "attr", "required": False, "dtype": "str", "value": "update"},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -2},
        ]
    },
    {
        "inputs": [
            {"name": "var_list", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 8, 16]},
            {"name": "indice", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 8, 16]},
            {"name": "mask", "type": "tensor", "required": False, "dtype": "uint8", "shape": [4]},
            {"name": "reduce", "type": "attr", "required": False, "dtype": "str", "value": "add"},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -2},
        ]
    },
    {
        "inputs": [
            {"name": "var_list", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16, 32, 64]},
            {"name": "indice", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16, 32, 64]},
            {"name": "mask", "type": "tensor", "required": False, "dtype": "uint8", "shape": [16]},
            {"name": "reduce", "type": "attr", "required": False, "dtype": "str", "value": "update"},
            {"name": "axis", "type": "attr", "required": False, "dtype": "int", "value": -2},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        var_shape = None
        for item in case["inputs"]:
            if item["name"] == "var_list":
                var_shape = item["shape"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "indice":
                    n = item["shape"][0]
                    max_idx = var_shape[-2]
                    t = torch.randint(0, max_idx, (n,)).to(dtype)
                elif item["name"] == "mask":
                    n = item["shape"][0]
                    t = torch.ones(n, dtype=dtype)
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
