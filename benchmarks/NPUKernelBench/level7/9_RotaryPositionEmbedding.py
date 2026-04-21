import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs RotaryPositionEmbedding.
    Single-path rotary position encoding.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: int):
        """
        Args:
            x: Input tensor, 4D.
            cos: Cosine tensor.
            sin: Sine tensor.
            mode: 0=half, 1=interleave, 2=quarter, 3=interleave-half.
        Returns:
            out: Rotary position encoded tensor.
        """
        if mode == 0:  # half
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            x_rotate = torch.cat((-x2, x1), dim=-1)
            y = x * cos + x_rotate * sin
        elif mode == 1:  # interleave
            x1 = x[..., ::2].view(-1, 1)
            x2 = x[..., 1::2].view(-1, 1)
            x_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape)
            y = x * cos + x_rotate * sin
        elif mode == 2:  # quarter
            x1 = x[..., : x.shape[-1] // 4]
            x2 = x[..., x.shape[-1] // 4: x.shape[-1] // 2]
            x3 = x[..., x.shape[-1] // 2: x.shape[-1] // 4 * 3]
            x4 = x[..., x.shape[-1] // 4 * 3:]
            x_rotate = torch.cat((-x2, x1, -x4, x3), dim=-1)
            y = x * cos + x_rotate * sin
        elif mode == 3:  # interleave-half
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            x_part1 = torch.cat((x1, x2), dim=-1)
            x_part2 = torch.cat((-x2, x1), dim=-1)
            y = x_part1 * cos + x_part2 * sin
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return y


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_RotaryPositionEmbedding.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
    }

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        group = []
        for inp in inputs:
            if inp["type"] == "tensor":
                dtype = dtype_map[inp["dtype"]]
                if inp["dtype"] in ("float32", "float16", "bfloat16"):
                    if "cos" in inp["name"]:
                        t = torch.randn(inp["shape"], dtype=torch.float32)
                        group.append(torch.cos(t).to(dtype))
                    elif "sin" in inp["name"]:
                        t = torch.randn(inp["shape"], dtype=torch.float32)
                        group.append(torch.sin(t).to(dtype))
                    else:
                        group.append(torch.randn(inp["shape"], dtype=dtype))
                else:
                    group.append(torch.randint(0, 10, inp["shape"], dtype=dtype))
            elif inp["type"] == "attr":
                group.append(inp["value"])
        input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
