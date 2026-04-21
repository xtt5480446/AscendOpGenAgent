import torch
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
