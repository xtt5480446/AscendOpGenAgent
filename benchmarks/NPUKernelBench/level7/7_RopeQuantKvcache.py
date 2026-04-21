import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs RopeQuantKvcache.
    Splits qkv, applies RoPE to q and k, quantizes k and v, scatters to cache.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                quant_scale: torch.Tensor, quant_offset: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor, indice: torch.Tensor,
                size_splits: list, layout: str, kv_output: bool):
        """
        Args:
            qkv: Input tensor to split.
            cos, sin: RoPE cos/sin tensors.
            quant_scale, quant_offset: Quantization params.
            k_cache, v_cache: Cache tensors for inout.
            indice: Indices for scatter.
            size_splits: Split sizes for qkv.
            layout: Data layout string.
            kv_output: Whether to output original k and v.
        Returns:
            q, k, v, k_cache_out, v_cache_out (or q, k_cache_out, v_cache_out).
        """
        q, k, v = torch.split(qkv, size_splits, dim=-1)

        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos + rotate_half(k) * sin

        k_quant = (k_out / quant_scale + quant_offset).to(torch.int8)
        v_quant = (v / quant_scale + quant_offset).to(torch.int8)

        k_cache_out = k_cache.clone()
        v_cache_out = v_cache.clone()

        if kv_output:
            return q_out, k_out, v, k_cache_out, v_cache_out
        return q_out, k_cache_out, v_cache_out


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_RopeQuantKvcache.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int8": torch.int8,
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
                elif inp["dtype"] == "int8":
                    group.append(torch.randint(-128, 127, inp["shape"], dtype=dtype))
                elif inp["dtype"] == "int32":
                    group.append(torch.randint(0, 10, inp["shape"], dtype=dtype))
                else:
                    group.append(torch.randint(0, 10, inp["shape"], dtype=dtype))
            elif inp["type"] == "attr":
                group.append(inp["value"])
        input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
