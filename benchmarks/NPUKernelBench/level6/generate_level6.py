#!/usr/bin/env python3
"""Generate level6 benchmark files for a3_conversion operators."""

import json
import os

OUT_DIR = "/tmp/AscendOpGenAgent/benchmarks/NPUKernelBench/level6"

DTYPE_MAP = {
    "float32": "torch.float32",
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "int8": "torch.int8",
    "uint8": "torch.uint8",
    "bool": "torch.bool",
}

def write_py(name, id_num, content):
    path = os.path.join(OUT_DIR, f"{id_num}_{name}.py")
    with open(path, "w") as f:
        f.write(content)

def write_json(name, id_num, cases):
    path = os.path.join(OUT_DIR, f"{id_num}_{name}.json")
    with open(path, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

# =============================================================================
# 1. ChunkCat
# =============================================================================
write_py("ChunkCat", 1, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """ChunkCat: chunk tensors along dim into num_chunks then cat along next dim."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, dim: int, num_chunks: int) -> torch.Tensor:
        chunks = []
        for t in tensors:
            chunks.extend(torch.chunk(t, num_chunks, dim=dim))
        cat_dim = dim + 1 if dim + 1 < tensors[0].dim() else dim
        return torch.cat(chunks, dim=cat_dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_ChunkCat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        tensors_info = inputs[0]
        dtype = dtype_map[tensors_info["dtype"]]
        shapes = tensors_info["shapes"]
        tensors = [torch.randn(shape, dtype=dtype) for shape in shapes]
        dim = inputs[1]["value"]
        num_chunks = inputs[2]["value"]
        input_groups.append([tensors, dim, num_chunks])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("ChunkCat", 1, [
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float32", "shapes": [[4, 8, 16], [4, 8, 16]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 1}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float16", "shapes": [[2, 16, 32], [2, 16, 32]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 1}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 4}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "bfloat16", "shapes": [[8, 32, 64]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float32", "shapes": [[1, 64, 128], [1, 64, 128], [1, 64, 128]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 1}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float16", "shapes": [[16, 256]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 0}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 4}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "bfloat16", "shapes": [[3, 12, 24, 48], [3, 12, 24, 48]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 3}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float32", "shapes": [[32, 64]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 0}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float16", "shapes": [[2, 4, 8, 16, 32]]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 3}, {"name": "num_chunks", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
])

# =============================================================================
# 2. CircularPad
# =============================================================================
write_py("CircularPad", 2, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """CircularPad: pad tensor with circular padding."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        return torch.nn.functional.pad(x, pad, mode='circular')


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_CircularPad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int32": torch.int32,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("CircularPad", 2, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 3, 32, 32]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3, 64, 64]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [2, 16, 56, 56]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 3, 4]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 64, 28, 28]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 1, 1]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int8", "shape": [1, 3, 16, 16]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 8, 16]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 128, 14, 14]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [3, 3, 3, 3]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 32, 32]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 1, 3, 2]}]},
])

# =============================================================================
# 3. CircularPadGrad
# =============================================================================
write_py("CircularPadGrad", 3, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """CircularPadGrad: backward of circular padding."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        grad_input = torch.nn.functional.pad(grad_output, pad, mode='circular')
        return grad_input


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_CircularPadGrad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("CircularPadGrad", 3, [
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 3, 34, 34]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3, 68, 68]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [2, 16, 60, 62]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 3, 4]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 64, 30, 30]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3, 20, 20]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 18]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 128, 20, 20]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [3, 3, 3, 3]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 32, 36]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 1, 3, 2]}]},
])

# =============================================================================
# 4. CoalesceSparse
# =============================================================================
write_py("CoalesceSparse", 4, '''import torch
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
''')

write_json("CoalesceSparse", 4, [
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int64", "shape": [2, 6], "value": [[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [6], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 8], "value": [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "float16", "shape": [8], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int64", "shape": [3, 10], "value": [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [10], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [2, 3, 2]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 4], "value": [[0, 1, 0, 1], [0, 0, 1, 1]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "int32", "shape": [4], "value": [1, 2, 3, 4]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int64", "shape": [2, 12], "value": [[0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "float16", "shape": [12], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [3, 3]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int64", "shape": [2, 5], "value": [[0, 0, 1, 1, 2], [0, 1, 0, 1, 0]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [5], "value": [1.0, 2.0, 3.0, 4.0, 5.0]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [3, 2]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [3, 6], "value": [[0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "int32", "shape": [6], "value": [1, 2, 3, 4, 5, 6]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2]}]},
    {"inputs": [{"name": "indices", "type": "tensor", "required": True, "dtype": "int64", "shape": [2, 3], "value": [[0, 1, 2], [0, 1, 2]]}, {"name": "values", "type": "tensor", "required": True, "dtype": "float32", "shape": [3], "value": [1.0, 2.0, 3.0]}, {"name": "shape", "type": "attr", "required": True, "dtype": "list", "value": [3, 3]}]},
])

# =============================================================================
# 5. ConcatDV2
# =============================================================================
write_py("ConcatDV2", 5, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """ConcatDV2: concatenate tensors along a dimension."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, concat_dim: int = 0) -> torch.Tensor:
        return torch.cat(tensors, dim=concat_dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_ConcatDV2.json")
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
            "bool": torch.bool,
        }
        tensors_info = inputs[0]
        dtype = dtype_map[tensors_info["dtype"]]
        shapes = tensors_info["shapes"]
        tensors = [torch.randn(shape, dtype=dtype) for shape in shapes]
        concat_dim = inputs[1]["value"]
        input_groups.append([tensors, concat_dim])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("ConcatDV2", 5, [
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float32", "shapes": [[4, 16], [4, 16]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float16", "shapes": [[2, 8, 32], [2, 8, 32], [2, 8, 32]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "bfloat16", "shapes": [[1, 64, 64], [1, 64, 64]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "int32", "shapes": [[8, 16], [8, 16], [8, 16]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float32", "shapes": [[16, 32, 64]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "bool", "shapes": [[2, 4, 8, 16], [2, 4, 8, 16]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 3}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "int64", "shapes": [[32, 64], [32, 64]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float16", "shapes": [[1, 3, 224, 224], [1, 3, 224, 224]]}, {"name": "concat_dim", "type": "attr", "required": True, "dtype": "int", "value": 0}]},
])

# =============================================================================
# 6. Contiguous
# =============================================================================
write_py("Contiguous", 6, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """Contiguous: make a non-contiguous tensor contiguous."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_Contiguous.json")
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
            "bool": torch.bool,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        if x_info.get("non_contiguous", False):
            x = x.transpose(0, -1)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("Contiguous", 6, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 64], "non_contiguous": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 32, 64], "non_contiguous": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 16, 32, 64], "non_contiguous": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [128, 256], "non_contiguous": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 16, 32], "non_contiguous": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bool", "shape": [64, 128], "non_contiguous": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int64", "shape": [2, 4, 8, 16, 32], "non_contiguous": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3, 224, 224], "non_contiguous": False}]},
])

# =============================================================================
# 7. DiagFlat
# =============================================================================
write_py("DiagFlat", 7, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """DiagFlat: create a diagonal matrix from a flat tensor."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
        return torch.diagflat(x, offset=diagonal)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_DiagFlat.json")
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
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        diagonal = inputs[1]["value"]
        input_groups.append([x, diagonal])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("DiagFlat", 7, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [4]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [8]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": -1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 3]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [3, 4]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int64", "shape": [5]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": -2}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 2, 2]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [10]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 0}]},
])

# =============================================================================
# 8. DiagV2
# =============================================================================
write_py("DiagV2", 8, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """DiagV2: extract diagonal elements from a 2D tensor."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
        return torch.diagonal(x, offset=diagonal)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_DiagV2.json")
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
            "bool": torch.bool,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        diagonal = inputs[1]["value"]
        input_groups.append([x, diagonal])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("DiagV2", 8, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 8]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 8]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 16]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": -1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 4]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 32]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": -2}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int64", "shape": [10, 10]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 0}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bool", "shape": [5, 5]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [64, 32]}, {"name": "diagonal", "type": "attr", "required": False, "dtype": "int", "value": -1}]},
])

# =============================================================================
# 9. FeedsRepeat
# =============================================================================
write_py("FeedsRepeat", 9, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """FeedsRepeat: repeat feeds rows according to repeat times."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        repeat_times = feeds_repeat_times.tolist()
        repeated = []
        for i, t in enumerate(repeat_times):
            repeated.append(feeds[i:i+1].repeat(int(t), 1))
        result = torch.cat(repeated, dim=0)
        if result.shape[0] < output_feeds_size:
            pad = torch.zeros(output_feeds_size - result.shape[0], result.shape[1], dtype=result.dtype, device=result.device)
            result = torch.cat([result, pad], dim=0)
        return result[:output_feeds_size]


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_FeedsRepeat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        feeds_info = inputs[0]
        dtype = dtype_map[feeds_info["dtype"]]
        feeds = torch.randn(feeds_info["shape"], dtype=dtype)
        times_dtype_map = {"int32": torch.int32, "int64": torch.int64}
        times_info = inputs[1]
        times_dtype = times_dtype_map[times_info["dtype"]]
        feeds_repeat_times = torch.tensor(times_info["value"], dtype=times_dtype)
        output_feeds_size = inputs[2]["value"]
        input_groups.append([feeds, feeds_repeat_times, output_feeds_size])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("FeedsRepeat", 9, [
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 16]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int32", "shape": [4], "value": [1, 2, 1, 3]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 10}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 32]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int64", "shape": [2], "value": [2, 3]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 8}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [3, 64]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int32", "shape": [3], "value": [1, 1, 1]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 5}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "float32", "shape": [5, 128]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int64", "shape": [5], "value": [2, 2, 2, 2, 2]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 12}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 256]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int32", "shape": [1], "value": [5]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 5}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [6, 8]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int64", "shape": [6], "value": [1, 0, 2, 1, 3, 1]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 15}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 512]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int32", "shape": [8], "value": [1, 1, 1, 1, 1, 1, 1, 1]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 8}]},
    {"inputs": [{"name": "feeds", "type": "tensor", "required": True, "dtype": "float16", "shape": [4, 1024]}, {"name": "feeds_repeat_times", "type": "tensor", "required": True, "dtype": "int64", "shape": [4], "value": [3, 1, 2, 1]}, {"name": "output_feeds_size", "type": "attr", "required": True, "dtype": "int", "value": 10}]},
])

# =============================================================================
# 10. Fill
# =============================================================================
write_py("Fill", 10, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """Fill: fill a tensor with a scalar value."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dims: list, value: float) -> torch.Tensor:
        return torch.full(dims, value)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "10_Fill.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dims = inputs[0]["value"]
        value = inputs[1]["value"]
        input_groups.append([dims, value])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("Fill", 10, [
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [16, 32]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": 1.0}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [8, 16, 32]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": 0.0}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [4, 8, 16, 32]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": -1.0}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [128]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": 3.14}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [2, 4, 8, 16, 32]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": 2.0}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [64, 64]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": 0.5}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [1, 3, 224, 224]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": 1.0}]},
    {"inputs": [{"name": "dims", "type": "attr", "required": True, "dtype": "list", "value": [32, 64, 128]}, {"name": "value", "type": "attr", "required": True, "dtype": "float", "value": -0.5}]},
])

# =============================================================================
# 11. FillDiagonalV2
# =============================================================================
write_py("FillDiagonalV2", 11, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """FillDiagonalV2: fill diagonal of a tensor with a value."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fill_value: float, wrap: bool = False) -> torch.Tensor:
        x = x.clone()
        torch.fill_diagonal_(x, fill_value, wrap=wrap)
        return x


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "11_FillDiagonalV2.json")
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
            "bool": torch.bool,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        fill_value = inputs[1]["value"]
        wrap = inputs[2]["value"]
        input_groups.append([x, fill_value, wrap])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("FillDiagonalV2", 11, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 8]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 1.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 8]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 0.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 16]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": -1.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 4]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 5.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 32]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 2.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int64", "shape": [10, 10]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 0.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": True}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bool", "shape": [5, 5]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 1.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": False}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [6, 4]}, {"name": "fill_value", "type": "attr", "required": True, "dtype": "float", "value": 3.0}, {"name": "wrap", "type": "attr", "required": False, "dtype": "bool", "value": True}]},
])

# =============================================================================
# 12. Flatten
# =============================================================================
write_py("Flatten", 12, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """Flatten: flatten a tensor to 2D starting from a given axis."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, axis: int = 1) -> torch.Tensor:
        return torch.flatten(x, start_dim=axis)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "12_Flatten.json")
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
            "bool": torch.bool,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        axis = inputs[1]["value"]
        input_groups.append([x, axis])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("Flatten", 12, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 3, 4, 5]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3, 224, 224]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [4, 8, 16, 32]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [16, 32]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 4, 8, 16, 32]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "int64", "shape": [8, 16, 32]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bool", "shape": [4, 4, 4, 4]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [32, 64, 128]}, {"name": "axis", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
])

# =============================================================================
# 13. MatMulV2CompressDequant
# =============================================================================
write_py("MatMulV2CompressDequant", 13, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """MatMulV2CompressDequant: matrix multiply with dequantization."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor = None, deq_scale: float = 1.0) -> torch.Tensor:
        result = torch.matmul(x1.float(), x2.float())
        if bias is not None:
            result = result + bias.float()
        result = result * deq_scale
        return result.to(torch.float16)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "13_MatMulV2CompressDequant.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1 = torch.randint(-128, 127, inputs[0]["shape"], dtype=torch.int8)
        x2 = torch.randint(-128, 127, inputs[1]["shape"], dtype=torch.int8)
        bias = torch.randint(-1000, 1000, inputs[2]["shape"], dtype=torch.int32) if inputs[2] is not None else None
        deq_scale = inputs[3]["value"]
        input_groups.append([x1, x2, bias, deq_scale])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("MatMulV2CompressDequant", 13, [
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [32, 64]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [64, 128]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [128]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 0.5}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [16, 32]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [32, 64]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [64]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 1.0}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [64, 128]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [128, 256]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [256]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 0.25}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [8, 16]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [16, 32]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [32]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 2.0}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [128, 256]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [256, 512]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [512]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 0.125}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [4, 8]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [8, 16]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [16]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 1.0}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [256, 512]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [512, 1024]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [1024]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 0.0625}]},
    {"inputs": [{"name": "x1", "type": "tensor", "required": True, "dtype": "int8", "shape": [1, 64]}, {"name": "x2", "type": "tensor", "required": True, "dtype": "int8", "shape": [64, 128]}, {"name": "bias", "type": "tensor", "required": False, "dtype": "int32", "shape": [128]}, {"name": "deq_scale", "type": "attr", "required": False, "dtype": "float", "value": 0.5}]},
])

# =============================================================================
# 14. PadV3GradReplicate
# =============================================================================
write_py("PadV3GradReplicate", 14, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """PadV3GradReplicate: backward of replication padding (1d/2d)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, self_shape: list, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        if len(paddings) == 2:
            grad_input = torch.nn.functional.pad(grad_output, pad, mode='replicate')
        else:
            grad_input = torch.nn.functional.pad(grad_output, pad, mode='replicate')
        return grad_input


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "14_PadV3GradReplicate.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("PadV3GradReplicate", 14, [
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 18]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 4, 36, 36]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [1, 3, 68, 68]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 3, 4]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 32]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 16, 32, 32]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 64]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [3, 3]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [4, 8, 16, 16]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [1, 64, 30, 30]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}]},
])

# =============================================================================
# 15. PadV3GradReplication
# =============================================================================
write_py("PadV3GradReplication", 15, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """PadV3GradReplication: backward of replication padding 3d."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        grad_input = torch.nn.functional.pad(grad_output, pad, mode='replicate')
        return grad_input


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "15_PadV3GradReplication.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("PadV3GradReplication", 15, [
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 4, 8, 12, 12]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 2, 4, 16, 16]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [1, 1, 3, 10, 10]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 1, 2, 1, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 16, 8, 8]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 2, 2, 6, 6]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 4, 8, 14, 14]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [2, 4, 4, 8, 8]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 2, 2, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 2, 8, 12, 12]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]}]},
])

# =============================================================================
# 16. PadV4Grad
# =============================================================================
write_py("PadV4Grad", 16, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """PadV4Grad: backward of reflection padding (1d/2d)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        grad_input = torch.nn.functional.pad(grad_output, pad, mode='reflect')
        return grad_input


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "16_PadV4Grad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("PadV4Grad", 16, [
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 18]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 4, 36, 36]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [1, 3, 68, 68]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 3, 4]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 32]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 16, 32, 32]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 64]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [3, 3]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [4, 8, 16, 16]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [1, 64, 30, 30]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}]},
])

# =============================================================================
# 17. ReflectionPad3dGrad
# =============================================================================
write_py("ReflectionPad3dGrad", 17, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """ReflectionPad3dGrad: backward of reflection padding 3d."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, paddings: list) -> torch.Tensor:
        pad = tuple(paddings)
        grad_input = torch.nn.functional.pad(grad_output, pad, mode='reflect')
        return grad_input


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "17_ReflectionPad3dGrad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        paddings = inputs[1]["value"]
        input_groups.append([x, paddings])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("ReflectionPad3dGrad", 17, [
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 4, 8, 12, 12]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 2, 4, 16, 16]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [1, 1, 3, 10, 10]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 1, 2, 1, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 16, 8, 8]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 2, 2, 6, 6]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 4, 8, 14, 14]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2, 2, 2, 2]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [2, 4, 4, 8, 8]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 2, 2, 1, 1]}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 2, 8, 12, 12]}, {"name": "paddings", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1, 1, 1]}]},
])

# =============================================================================
# 18. StackBallQuery
# =============================================================================
write_py("StackBallQuery", 18, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """StackBallQuery: find points within a radius for each center point."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, xyz: torch.Tensor, center_xyz: torch.Tensor, max_radius: float, sample_num: int) -> torch.Tensor:
        batch_size = center_xyz.shape[0]
        idx = torch.zeros(batch_size, center_xyz.shape[1], sample_num, dtype=torch.int32)
        for b in range(batch_size):
            for i in range(center_xyz.shape[1]):
                center = center_xyz[b, i]
                dists = torch.norm(xyz[b] - center.unsqueeze(0), dim=1)
                mask = dists < max_radius
                valid_indices = torch.where(mask)[0]
                if valid_indices.numel() >= sample_num:
                    idx[b, i] = valid_indices[:sample_num].to(torch.int32)
                else:
                    n = valid_indices.numel()
                    if n > 0:
                        idx[b, i, :n] = valid_indices[:n].to(torch.int32)
        return idx


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "18_StackBallQuery.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
        }
        xyz_info = inputs[0]
        dtype = dtype_map[xyz_info["dtype"]]
        xyz = torch.randn(xyz_info["shape"], dtype=dtype)
        center_xyz = torch.randn(inputs[1]["shape"], dtype=dtype)
        max_radius = inputs[2]["value"]
        sample_num = inputs[3]["value"]
        input_groups.append([xyz, center_xyz, max_radius, sample_num])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("StackBallQuery", 18, [
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 100, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 16, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 1.0}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 16}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 256, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 32, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 2.0}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 32}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 64, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 0.5}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 8}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 512, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 64, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 1.5}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 32}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 128, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [1, 16, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 3.0}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 16}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 32, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 4, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 0.25}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 4}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 200, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 20, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 1.0}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 16}]},
    {"inputs": [{"name": "xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 1024, 3]}, {"name": "center_xyz", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 128, 3]}, {"name": "max_radius", "type": "attr", "required": True, "dtype": "float", "value": 2.0}, {"name": "sample_num", "type": "attr", "required": True, "dtype": "int", "value": 64}]},
])

# =============================================================================
# 19. StridedSliceAssignV2
# =============================================================================
write_py("StridedSliceAssignV2", 19, '''import torch
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
''')

write_json("StridedSliceAssignV2", 19, [
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 16]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [4, 8]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [1, 1]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": None}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 32, 64]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 16, 32]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [8, 16, 32]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": None}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [4, 8, 16, 32]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [2, 4, 8, 16]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [2, 4, 8, 16]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1, 1]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": None}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "int32", "shape": [8, 8]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 4]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [4, 4]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": None}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 64]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 32]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [16, 32]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [2, 2]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": [0, 1]}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 16, 16]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 8, 8]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [8, 8, 8]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": None}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "int64", "shape": [4, 4, 4]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "int64", "shape": [2, 2, 2]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [2, 2, 2]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [1, 1, 1]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": [0, 1, 2]}]},
    {"inputs": [{"name": "var", "type": "tensor", "required": True, "dtype": "float32", "shape": [64, 128]}, {"name": "input_value", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 64]}, {"name": "begin", "type": "attr", "required": True, "dtype": "list", "value": [0, 0]}, {"name": "end", "type": "attr", "required": True, "dtype": "list", "value": [32, 64]}, {"name": "strides", "type": "attr", "required": True, "dtype": "list", "value": [1, 1]}, {"name": "axes", "type": "attr", "required": False, "dtype": "list", "value": None}]},
])

# =============================================================================
# 20. TransposeV2
# =============================================================================
write_py("TransposeV2", 20, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """TransposeV2: permute tensor dimensions."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, perm: list) -> torch.Tensor:
        return torch.permute(x, perm)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "20_TransposeV2.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        perm = inputs[1]["value"]
        input_groups.append([x, perm])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("TransposeV2", 20, [
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 3, 4]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [2, 0, 1]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 3, 224, 224]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [0, 2, 3, 1]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [4, 8, 16, 32]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [3, 2, 1, 0]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 32]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [1, 0]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [2, 4, 8, 16, 32]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [4, 3, 2, 1, 0]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 16, 32]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [1, 2, 0]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [32, 64, 128]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [2, 0, 1]}]},
    {"inputs": [{"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [1, 64, 56, 56]}, {"name": "perm", "type": "attr", "required": True, "dtype": "list", "value": [0, 1, 3, 2]}]},
])

# =============================================================================
# 21. UnfoldGrad
# =============================================================================
write_py("UnfoldGrad", 21, '''import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """UnfoldGrad: backward of unfold operation."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, input_sizes: list, dim: int, size: int, step: int) -> torch.Tensor:
        return torch.ops.aten.unfold_backward(grad_output, input_sizes, dim, size, step)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "21_UnfoldGrad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        grad_output = torch.randn(x_info["shape"], dtype=dtype)
        input_sizes = inputs[1]["value"]
        dim = inputs[2]["value"]
        size = inputs[3]["value"]
        step = inputs[4]["value"]
        input_groups.append([grad_output, input_sizes, dim, size, step])
    return input_groups


def get_init_inputs():
    return []
''')

write_json("UnfoldGrad", 21, [
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [6, 2]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [7]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 0}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [3, 2]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [7]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 0}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 16, 4]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [8, 16]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 1}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 4}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 8, 8, 3]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [4, 8, 10]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 3}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [16, 4]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [16]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 0}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 4}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 4, 4, 2]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [2, 4, 5]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 2}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 1}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [32, 8, 4]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [32, 10]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 1}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 4}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 2}]},
    {"inputs": [{"name": "grad_output", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 3]}, {"name": "input_sizes", "type": "attr", "required": True, "dtype": "list", "value": [10]}, {"name": "dim", "type": "attr", "required": True, "dtype": "int", "value": 0}, {"name": "size", "type": "attr", "required": True, "dtype": "int", "value": 3}, {"name": "step", "type": "attr", "required": True, "dtype": "int", "value": 3}]},
])

print("Done! Generated 21 operator benchmark files in", OUT_DIR)
