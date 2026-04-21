import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs InterleaveRope.
    Applies interleaved rotary position encoding to input x.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Args:
            x: Input tensor, shape (B, N, S, D), D must be 64.
            cos: Cosine tensor, shape (B, N, S, D), N must be 1.
            sin: Sine tensor, shape (B, N, S, D), N must be 1.
        Returns:
            y: Rotary position encoded tensor, same shape as x.
        """
        # Reshape and transpose: equivalent to interleaving pairs
        q = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).transpose(-1, -2).reshape(*x.shape)
        # RotateHalf
        q1 = q[..., :q.shape[-1] // 2]
        q2 = q[..., q.shape[-1] // 2:]
        q_rot = torch.cat((-q2, q1), dim=-1)
        y = q * cos + q_rot * sin
        return y


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_InterleaveRope.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        group = []
        for inp in inputs:
            if inp["type"] == "tensor":
                dtype = dtype_map[inp["dtype"]]
                if "cos" in inp["name"]:
                    t = torch.randn(inp["shape"], dtype=torch.float32)
                    group.append(torch.cos(t).to(dtype))
                elif "sin" in inp["name"]:
                    t = torch.randn(inp["shape"], dtype=torch.float32)
                    group.append(torch.sin(t).to(dtype))
                else:
                    group.append(torch.randn(inp["shape"], dtype=dtype))
            elif inp["type"] == "attr":
                group.append(inp["value"])
        input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
