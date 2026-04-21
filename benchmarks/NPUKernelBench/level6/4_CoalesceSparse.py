import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """CoalesceSparse: coalesce sparse tensor indices and values."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, indices: torch.Tensor, values: torch.Tensor, shape: list) -> torch.Tensor:
        sparse = torch.sparse_coo_tensor(indices, values, shape)
        coalesced = sparse.coalesce()
        return coalesced.values()


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_CoalesceSparse.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        idx_dtype_map = {
            "int32": torch.int32,
            "int64": torch.int64,
        }
        indices_info = inputs[0]
        values_info = inputs[1]
        idx_dtype = idx_dtype_map[indices_info["dtype"]]
        val_dtype = dtype_map[values_info["dtype"]]
        indices = torch.tensor(indices_info["value"], dtype=idx_dtype)
        values = torch.tensor(values_info["value"], dtype=val_dtype)
        shape = inputs[2]["value"]
        input_groups.append([indices, values, shape])
    return input_groups


def get_init_inputs():
    return []
