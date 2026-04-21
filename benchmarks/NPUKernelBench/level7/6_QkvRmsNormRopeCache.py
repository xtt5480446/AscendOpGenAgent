import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs QkvRmsNormRopeCache.
    Simplified: no quantization, no scatter to cache.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, qkv: torch.Tensor, q_gamma: torch.Tensor, k_gamma: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, index: torch.Tensor,
                q_out: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                qkv_size: list, head_nums: list, epsilon: float,
                cache_mode: str, is_output_qkv: bool):
        """
        Args:
            qkv: Input tensor, shape (B*S, N_qkv*D).
            q_gamma, k_gamma: Gamma for RMSNorm, shape (D,).
            cos, sin: Cos/Sin for RoPE.
            index: Index tensor.
            q_out, k_cache, v_cache: Inout cache tensors.
            qkv_size: [B, S, N_qkv, D].
            head_nums: [N_q, N_k, N_v].
            epsilon: Epsilon for RMSNorm.
            cache_mode: Cache mode.
            is_output_qkv: Whether to output pre-quant values.
        Returns:
            q_out, k_cache, v_cache (and optional pre-quant outputs).
        """
        B, S, N_qkv, D = qkv_size
        N_q, N_k, N_v = head_nums

        # SplitVD
        q = qkv[..., :N_q * D]
        k = qkv[..., N_q * D:(N_q + N_k) * D]
        v = qkv[..., (N_q + N_k) * D:]

        # Reshape to (B*S, N, D)
        q = q.view(B * S, N_q, D)
        k = k.view(B * S, N_k, D)
        v = v.view(B * S, N_v, D)

        # RMSNorm
        def rms_norm(x, gamma):
            ms = (x ** 2).mean(dim=-1, keepdim=True)
            rms = torch.sqrt(ms + epsilon)
            return (x / rms) * gamma

        q_norm = rms_norm(q, q_gamma)
        k_norm = rms_norm(k, k_gamma)

        # RoPE half
        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rope = q_norm * cos + rotate_half(q_norm) * sin
        k_rope = k_norm * cos + rotate_half(k_norm) * sin

        if is_output_qkv:
            return q_rope, k_rope, v, q_rope, k_rope, v
        return q_rope, k_rope, v


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_QkvRmsNormRopeCache.json")
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
