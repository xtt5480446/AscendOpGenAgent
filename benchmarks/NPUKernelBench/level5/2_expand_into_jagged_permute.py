import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ExpandIntoJaggedPermute: expands sparse permute indices from table dimension to batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permute: torch.Tensor, input_offset: torch.Tensor,
                output_offset: torch.Tensor, output_size: int) -> torch.Tensor:
        """
        Args:
            permute (torch.Tensor): [tables], table-level permutation indices.
            input_offset (torch.Tensor): [tables+1], exclusive prefix sums of table lengths.
            output_offset (torch.Tensor): [tables+1], exclusive prefix sums of permuted table lengths.
            output_size (int): total output length.
        Returns:
            torch.Tensor: [output_size], expanded permutation indices.
        """
        tables = permute.shape[0]
        out = torch.zeros(output_size, dtype=torch.int32)
        for i in range(tables):
            p = permute[i].item()
            length = output_offset[i + 1].item() - output_offset[i].item()
            start = output_offset[i].item()
            inp_start = input_offset[p].item()
            for j in range(length):
                out[start + j] = inp_start + j
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "permute", "type": "tensor", "required": True, "dtype": "int32", "shape": [3]},
            {"name": "input_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "output_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "output_size", "type": "attr", "required": True, "dtype": "int", "value": 6},
        ]
    },
    {
        "inputs": [
            {"name": "permute", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "input_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [9]},
            {"name": "output_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [9]},
            {"name": "output_size", "type": "attr", "required": True, "dtype": "int", "value": 128},
        ]
    },
    {
        "inputs": [
            {"name": "permute", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "input_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [17]},
            {"name": "output_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [17]},
            {"name": "output_size", "type": "attr", "required": True, "dtype": "int", "value": 512},
        ]
    },
    {
        "inputs": [
            {"name": "permute", "type": "tensor", "required": True, "dtype": "int32", "shape": [32]},
            {"name": "input_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [33]},
            {"name": "output_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [33]},
            {"name": "output_size", "type": "attr", "required": True, "dtype": "int", "value": 2048},
        ]
    },
    {
        "inputs": [
            {"name": "permute", "type": "tensor", "required": True, "dtype": "int32", "shape": [64]},
            {"name": "input_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [65]},
            {"name": "output_offset", "type": "tensor", "required": True, "dtype": "int32", "shape": [65]},
            {"name": "output_size", "type": "attr", "required": True, "dtype": "int", "value": 8192},
        ]
    },
]


_DTYPE_MAP = {
    "int32": torch.int32,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "permute":
                    n = item["shape"][0]
                    t = torch.randperm(n).to(dtype)
                elif "offset" in item["name"]:
                    n = item["shape"][0]
                    t = torch.cumsum(torch.randint(1, 10, (n,)), dim=0).to(dtype)
                    t = torch.cat([torch.zeros(1, dtype=dtype), t[:-1]])
                else:
                    t = torch.randint(0, 100, item["shape"]).to(dtype)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
