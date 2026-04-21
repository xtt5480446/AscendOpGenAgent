import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs ApplyRotaryPosEmb.
    Applies rotary position embedding to query and key tensors.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, layout: int, rotary_mode: str):
        """
        Args:
            query: Input query tensor, 4D.
            key: Input key tensor, 4D.
            cos: Cosine tensor for RoPE.
            sin: Sine tensor for RoPE.
            layout: Layout format (1=BSND, 2=SBND, 3=BNSD, 4=TND).
            rotary_mode: Rotation mode ('half', 'interleave', 'quarter').
        Returns:
            q_embed, k_embed: Rotary position embedded query and key.
        """
        if rotary_mode == "half":
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)
        elif rotary_mode == "interleave":
            def rotate_half(x):
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                return torch.cat((-x2, x1), dim=-1).view(x.shape)
        elif rotary_mode == "quarter":
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 4]
                x2 = x[..., x.shape[-1] // 4: x.shape[-1] // 2]
                x3 = x[..., x.shape[-1] // 2: x.shape[-1] // 4 * 3]
                x4 = x[..., x.shape[-1] // 4 * 3:]
                return torch.cat((-x2, x1, -x4, x3), dim=-1)
        else:
            raise ValueError(f"Unsupported rotary_mode: {rotary_mode}")

        q_embed = query * cos + rotate_half(query) * sin
        k_embed = key * cos + rotate_half(key) * sin
        return q_embed, k_embed


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_ApplyRotaryPosEmb.json")
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
                if "cos" in inp["name"] or "sin" in inp["name"]:
                    t = torch.randn(inp["shape"], dtype=torch.float32)
                    group.append(torch.cos(t).to(dtype) if "cos" in inp["name"] else torch.sin(t).to(dtype))
                else:
                    group.append(torch.randn(inp["shape"], dtype=dtype))
            elif inp["type"] == "attr":
                group.append(inp["value"])
        input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
