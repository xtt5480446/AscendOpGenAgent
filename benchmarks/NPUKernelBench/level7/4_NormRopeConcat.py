import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs NormRopeConcat.
    Applies Norm, RoPE, and Concat on query, key, and value.
    Simplified: encoder inputs are None, only main path is computed.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                encoderQuery, encoderKey, encoderValue,
                normQueryWeight, normQueryBias, normKeyWeight, normKeyBias,
                normAddedQueryWeight, normAddedQueryBias,
                normAddedKeyWeight, normAddedKeyBias,
                ropeSin: torch.Tensor, ropeCos: torch.Tensor,
                normType: int, normAddedType: int, ropeType: int,
                concatOrder: int, eps: float, isTraining: bool):
        """
        Args:
            query, key, value: Input tensors, shape (B, S, N, D).
            ropeSin, ropeCos: RoPE sin/cos tensors, shape (seqRope, D).
            normType: 0=NONE, 1=LAYER_NORM, 2=LAYER_NORM_AFFINE, 3=RMS_NORM, 4=RMS_NORM_AFFINE.
            ropeType: 0=NONE, 1=INTERLEAVE, 2=HALF.
        Returns:
            queryOutput, keyOutput, valueOutput.
        """
        def apply_norm(x, weight, bias, norm_type):
            if norm_type == 0:
                return x
            elif norm_type == 1:  # LayerNorm
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                rstd = 1.0 / torch.sqrt(var + eps)
                return (x - mean) * rstd
            elif norm_type == 2:  # LayerNormAffine
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                rstd = 1.0 / torch.sqrt(var + eps)
                return weight * (x - mean) * rstd + bias
            elif norm_type == 3:  # RmsNorm
                ms = (x ** 2).mean(dim=-1, keepdim=True)
                rms = 1.0 / torch.sqrt(ms + eps)
                return x * rms
            elif norm_type == 4:  # RmsNormAffine
                ms = (x ** 2).mean(dim=-1, keepdim=True)
                rms = 1.0 / torch.sqrt(ms + eps)
                return weight * x * rms
            return x

        hiddenState_q = apply_norm(query, normQueryWeight, normQueryBias, normType)
        hiddenState_k = apply_norm(key, normKeyWeight, normKeyBias, normType)
        hiddenState_v = value  # Value has no norm

        # Transpose to (B, N, S, D)
        hiddenState_q = hiddenState_q.transpose(1, 2)
        hiddenState_k = hiddenState_k.transpose(1, 2)
        hiddenState_v = hiddenState_v.transpose(1, 2)

        # RoPE
        def apply_rope(x, sin, cos, rope_type):
            if rope_type == 0:
                return x
            seq_len = x.shape[2]
            sin_slice = sin[:seq_len]
            cos_slice = cos[:seq_len]
            if rope_type == 1:  # Interleave
                xv = x.view(*x.shape[:-1], -1, 2)
                x1, x2 = xv[..., 0], xv[..., 1]
                rotated = torch.stack([-x2, x1], dim=-1).flatten(3)
                return x * cos_slice + rotated * sin_slice
            elif rope_type == 2:  # Half
                x1, x2 = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
                rotated = torch.cat([-x2, x1], dim=-1)
                return x * cos_slice + rotated * sin_slice
            return x

        queryOutput = apply_rope(hiddenState_q, ropeSin, ropeCos, ropeType)
        keyOutput = apply_rope(hiddenState_k, ropeSin, ropeCos, ropeType)
        valueOutput = hiddenState_v

        return queryOutput, keyOutput, valueOutput


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_NormRopeConcat.json")
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
                if inp["dtype"] in ("float32", "float16", "bfloat16"):
                    if "ropeSin" in inp["name"] or "ropeCos" in inp["name"]:
                        t = torch.randn(inp["shape"], dtype=torch.float32)
                        group.append(torch.sin(t).to(dtype) if "Sin" in inp["name"] else torch.cos(t).to(dtype))
                    else:
                        group.append(torch.randn(inp["shape"], dtype=dtype))
                else:
                    group.append(torch.randint(0, 10, inp["shape"], dtype=dtype))
            elif inp["type"] == "attr":
                group.append(inp["value"])
            elif inp["type"] == "none":
                group.append(None)
        input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
