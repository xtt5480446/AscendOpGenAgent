import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs NormRopeConcatGrad.
    Backward of NormRopeConcat. Simplified: no encoder inputs.
    """
    def __init__(self):
        super(Model, self).__init__()

    def _forward_impl(self, query, key, value, normQueryWeight, normQueryMean, normQueryRstd,
                      normKeyWeight, normKeyMean, normKeyRstd,
                      ropeSin, ropeCos, normType, ropeType):
        def apply_norm(x, weight, mean, rstd, norm_type):
            if norm_type == 0:
                return x
            elif norm_type == 1:
                return (x - mean) * rstd
            elif norm_type == 2:
                return weight * (x - mean) * rstd
            elif norm_type == 3:
                return x * rstd
            elif norm_type == 4:
                return weight * x * rstd
            return x

        hidden_q = apply_norm(query, normQueryWeight, normQueryMean, normQueryRstd, normType)
        hidden_k = apply_norm(key, normKeyWeight, normKeyMean, normKeyRstd, normType)

        hidden_q = hidden_q.transpose(1, 2)
        hidden_k = hidden_k.transpose(1, 2)
        hidden_v = value.transpose(1, 2)

        def apply_rope(x, sin, cos, rope_type):
            if rope_type == 0:
                return x
            seq_len = x.shape[2]
            sin_s = sin[:seq_len]
            cos_s = cos[:seq_len]
            if rope_type == 1:
                xv = x.view(*x.shape[:-1], -1, 2)
                x1, x2 = xv[..., 0], xv[..., 1]
                rotated = torch.stack([-x2, x1], dim=-1).flatten(3)
                return x * cos_s + rotated * sin_s
            elif rope_type == 2:
                x1, x2 = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
                rotated = torch.cat([-x2, x1], dim=-1)
                return x * cos_s + rotated * sin_s
            return x

        q_out = apply_rope(hidden_q, ropeSin, ropeCos, ropeType)
        k_out = apply_rope(hidden_k, ropeSin, ropeCos, ropeType)
        return q_out, k_out, hidden_v

    def forward(self, gradQueryOutput: torch.Tensor, gradKeyOutput: torch.Tensor,
                gradValueOutput: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, encoderQuery, encoderKey, encoderValue,
                normQueryWeight, normQueryMean, normQueryRstd,
                normKeyWeight, normKeyMean, normKeyRstd,
                normAddQueryWeight, normAddQueryMean, normAddQueryRstd,
                normAddKeyWeight, normAddKeyMean, normAddKeyRstd,
                ropeSin: torch.Tensor, ropeCos: torch.Tensor,
                normType: int, normAddedType: int, ropeType: int, concatOrder: int):
        q_req = query.detach().requires_grad_(True)
        k_req = key.detach().requires_grad_(True)
        v_req = value.detach().requires_grad_(True)

        q_out, k_out, v_out = self._forward_impl(
            q_req, k_req, v_req,
            normQueryWeight, normQueryMean, normQueryRstd,
            normKeyWeight, normKeyMean, normKeyRstd,
            ropeSin, ropeCos, normType, ropeType
        )

        q_out.backward(gradQueryOutput, retain_graph=True)
        k_out.backward(gradKeyOutput, retain_graph=True)
        v_out.backward(gradValueOutput, retain_graph=True)

        return q_req.grad, k_req.grad, v_req.grad


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_NormRopeConcatGrad.json")
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
