import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs KvRmsNormRopeCache.
    Splits kv into left half for RMSNorm and right half for RoPE,
    then outputs k_rope and c_kv.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, kv: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, index: torch.Tensor, epsilon: float,
                cache_mode: str, is_output_kv: bool):
        """
        Args:
            kv: Input tensor, shape (Bkv, N, Skv, D), N=1.
            gamma: Gamma for RMSNorm, shape (Dv,).
            cos: Cosine tensor for RoPE.
            sin: Sine tensor for RoPE.
            index: Index tensor for cache scatter.
            epsilon: Epsilon for RMSNorm.
            cache_mode: Cache mode string.
            is_output_kv: Whether to output k_rope and c_kv.
        Returns:
            k_rope, c_kv if is_output_kv else empty tensors.
        """
        D = kv.shape[-1]
        Dv = D // 2
        Dk = D - Dv

        # RMSNorm on left half
        x_norm = kv[..., :Dv]
        square_x = x_norm * x_norm
        mean_square = square_x.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + epsilon)
        c_kv = (x_norm / rms) * gamma[:Dv]

        # Interleave RoPE on right half
        x_rope = kv[..., Dv:]
        x1 = x_rope[..., ::2]
        x2 = x_rope[..., 1::2]
        x_part1 = torch.cat((x1, x2), dim=-1)
        x_part2 = torch.cat((-x2, x1), dim=-1)
        k_rope = x_part1 * cos + x_part2 * sin

        if is_output_kv:
            return k_rope, c_kv
        else:
            return torch.empty_like(k_rope), torch.empty_like(c_kv)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_KvRmsNormRopeCache.json")
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
