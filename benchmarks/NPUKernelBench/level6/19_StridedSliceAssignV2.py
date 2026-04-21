import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """StridedSliceAssignV2: assign values to a strided slice of a tensor."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, input_value: torch.Tensor, begin: list, end: list, strides: list, axes: list = None) -> torch.Tensor:
        var = var.clone()
        slices = []
        for i in range(len(begin)):
            slices.append(slice(begin[i], end[i], strides[i]))
        if axes is not None:
            slices = [slices[axes.index(i)] if i in axes else slice(None) for i in range(var.dim())]
        var[tuple(slices)] = input_value
        return var


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "19_StridedSliceAssignV2.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        var_info = inputs[0]
        dtype = dtype_map[var_info["dtype"]]
        var = torch.randn(var_info["shape"], dtype=dtype)
        val_shape = inputs[1]["shape"]
        input_value = torch.randn(val_shape, dtype=dtype)
        begin = inputs[2]["value"]
        end = inputs[3]["value"]
        strides = inputs[4]["value"]
        axes = inputs[5]["value"] if len(inputs) > 5 else None
        input_groups.append([var, input_value, begin, end, strides, axes])
    return input_groups


def get_init_inputs():
    return []
