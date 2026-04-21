import torch
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
