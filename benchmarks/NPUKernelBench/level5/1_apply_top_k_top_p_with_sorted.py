import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ApplyTopKTopPWithSorted: applies top-k and top-p filtering on sorted values and indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, sorted_value: torch.Tensor, sorted_indices: torch.Tensor,
                p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sorted_value (torch.Tensor): [batch, vocab], sorted in ascending order.
            sorted_indices (torch.Tensor): [batch, vocab], original indices.
            p (torch.Tensor): [batch], top-p thresholds.
            k (torch.Tensor): [batch], top-k thresholds.
        Returns:
            torch.Tensor: [batch, vocab], filtered values restored to original order.
        """
        batch, vocab = sorted_value.shape
        out = torch.empty_like(sorted_value)
        for b in range(batch):
            kv = k[b].item()
            pv = p[b].item()
            sv = sorted_value[b].clone()
            # top-k filter
            if 1 <= kv <= vocab:
                topKValue = sv[vocab - kv]
                topKMask = sv < topKValue
                sv = torch.where(topKMask, torch.tensor(float('-inf'), dtype=sv.dtype, device=sv.device), sv)
            # softmax and cumsum for top-p
            probs = torch.softmax(sv, dim=-1)
            probs_sum = torch.cumsum(probs, dim=-1)
            topPMask = probs_sum <= (1.0 - pv)
            topPMask[-1] = False
            sv = torch.where(topPMask, torch.tensor(float('-inf'), dtype=sv.dtype, device=sv.device), sv)
            # restore order
            si = sorted_indices[b].long()
            out[b] = sv[si]
        return out


INPUT_CASES = [
    {
        "inputs": [
            {"name": "sorted_value", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 128]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 128]},
            {"name": "p", "type": "tensor", "required": True, "dtype": "float32", "shape": [4]},
            {"name": "k", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
        ]
    },
    {
        "inputs": [
            {"name": "sorted_value", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 1024]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [16, 1024]},
            {"name": "p", "type": "tensor", "required": True, "dtype": "float32", "shape": [16]},
            {"name": "k", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
        ]
    },
    {
        "inputs": [
            {"name": "sorted_value", "type": "tensor", "required": True, "dtype": "float16", "shape": [48, 8192]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [48, 8192]},
            {"name": "p", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "k", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
        ]
    },
    {
        "inputs": [
            {"name": "sorted_value", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 256]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [8, 256]},
            {"name": "p", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8]},
            {"name": "k", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
        ]
    },
    {
        "inputs": [
            {"name": "sorted_value", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 64]},
            {"name": "sorted_indices", "type": "tensor", "required": True, "dtype": "int32", "shape": [2, 64]},
            {"name": "p", "type": "tensor", "required": True, "dtype": "float32", "shape": [2]},
            {"name": "k", "type": "tensor", "required": True, "dtype": "int32", "shape": [2]},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] in ("sorted_indices", "k"):
                    t = torch.arange(0, 1)
                    for _ in range(len(item["shape"]) - 1):
                        t = t.unsqueeze(0)
                    t = t.expand(item["shape"]).contiguous()
                    if item["name"] == "sorted_indices":
                        batch = item["shape"][0]
                        vocab = item["shape"][1]
                        t = torch.stack([torch.randperm(vocab) for _ in range(batch)]).to(dtype)
                    else:
                        t = torch.randint(1, item["shape"][0] + 1, item["shape"]).to(dtype)
                elif item["name"] == "p":
                    t = torch.rand(item["shape"], dtype=torch.float32).clamp(0.1, 0.9)
                    if dtype != torch.float32:
                        t = t.to(dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
