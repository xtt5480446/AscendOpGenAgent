import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs RotaryPositionEmbeddingGrad.
    Backward of RotaryPositionEmbedding.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                xOptional: torch.Tensor, mode: int):
        """
        Args:
            dy: Gradient w.r.t. output, 4D.
            cos: Cosine tensor from forward.
            sin: Sine tensor from forward.
            xOptional: Forward input x (optional, for computing dcos/dsin).
            mode: 0=half, 1=interleave, 2=quarter, 3=interleave-half.
        Returns:
            dxOut, dcosOut, dsinOut.
        """
        if mode == 0:  # half
            dy1, dy2 = dy.chunk(2, dim=-1)
            cos1, cos2 = cos.chunk(2, dim=-1)
            sin1, sin2 = sin.chunk(2, dim=-1)
            x1, x2 = xOptional.chunk(2, dim=-1)
            dx = torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
            dcos = dy * xOptional
            dsin = dy * torch.cat((-x2, x1), dim=-1)
        elif mode == 1:  # interleave
            dy1, dy2 = dy[..., ::2], dy[..., 1::2]
            cos1, cos2 = cos[..., ::2], cos[..., 1::2]
            sin1, sin2 = sin[..., ::2], sin[..., 1::2]
            x1, x2 = xOptional[..., ::2], xOptional[..., 1::2]
            dx = torch.stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
            dcos = dy * xOptional
            dsin = dy * torch.stack((-x2, x1), dim=-1).reshape(dy.shape)
        elif mode == 2:  # quarter
            dy1, dy2, dy3, dy4 = dy.chunk(4, dim=-1)
            cos1, cos2, cos3, cos4 = cos.chunk(4, dim=-1)
            sin1, sin2, sin3, sin4 = sin.chunk(4, dim=-1)
            x1, x2, x3, x4 = xOptional.chunk(4, dim=-1)
            dx = torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1,
                            cos3 * dy3 + sin4 * dy4, cos4 * dy4 - sin3 * dy3), dim=-1)
            dcos = dy * xOptional
            dsin = dy * torch.cat((-x2, x1, -x4, x3), dim=-1)
        elif mode == 3:  # interleave-half
            dy1, dy2 = dy.chunk(2, dim=-1)
            cos1, cos2 = cos.chunk(2, dim=-1)
            sin1, sin2 = sin.chunk(2, dim=-1)
            x1, x2 = xOptional[..., ::2], xOptional[..., 1::2]
            dx = torch.stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
            dcos = dy * torch.cat((x1, x2), dim=-1)
            dsin = dy * torch.cat((-x2, x1), dim=-1)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return dx, dcos, dsin


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "10_RotaryPositionEmbeddingGrad.json")
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
