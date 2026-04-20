import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MaskedScatterWithPosition: copies updates to x at positions where mask is True.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, position: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor.
            mask (torch.Tensor): bool mask, broadcastable to x.
            position (torch.Tensor): prefix sum of mask before broadcast.
            updates (torch.Tensor): source values.
        Returns:
            torch.Tensor: updated x.
        """
        out = x.clone()
        m = mask.bool()
        # broadcast mask to x shape
        m_bc = m.expand_as(x)
        # position gives the index into updates for each True element
        pos = position.long()
        # simplified reference: for each True position, copy from updates using position
        # This is a simplified approximation for benchmark purposes
        flat_out = out.reshape(-1)
        flat_mask = m_bc.reshape(-1)
        flat_pos = pos.reshape(-1)
        flat_updates = updates.reshape(-1)
        true_count = flat_mask.sum().item()
        for i in range(flat_mask.numel()):
            if flat_mask[i]:
                p = flat_pos[i].item()
                if p < flat_updates.numel():
                    flat_out[i] = flat_updates[p]
        return flat_out.reshape_as(out)


INPUT_CASES = [
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 5]},
            {"name": "mask", "type": "tensor", "required": True, "dtype": "bool", "shape": [1, 5]},
            {"name": "position", "type": "tensor", "required": True, "dtype": "int64", "shape": [1, 5]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [2, 5]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 128]},
            {"name": "mask", "type": "tensor", "required": True, "dtype": "bool", "shape": [16, 128]},
            {"name": "position", "type": "tensor", "required": True, "dtype": "int64", "shape": [16, 128]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 128]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 256, 256]},
            {"name": "mask", "type": "tensor", "required": True, "dtype": "bool", "shape": [1, 256, 256]},
            {"name": "position", "type": "tensor", "required": True, "dtype": "int64", "shape": [1, 256, 256]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "float16", "shape": [8, 256, 256]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 64]},
            {"name": "mask", "type": "tensor", "required": True, "dtype": "bool", "shape": [4, 64]},
            {"name": "position", "type": "tensor", "required": True, "dtype": "int64", "shape": [4, 64]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "int32", "shape": [4, 64]},
        ]
    },
    {
        "inputs": [
            {"name": "x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [32, 32]},
            {"name": "mask", "type": "tensor", "required": True, "dtype": "bool", "shape": [32, 32]},
            {"name": "position", "type": "tensor", "required": True, "dtype": "int64", "shape": [32, 32]},
            {"name": "updates", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [32, 32]},
        ]
    },
]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def get_input_groups():
    groups = []
    for case in INPUT_CASES:
        inputs = []
        x_shape = None
        for item in case["inputs"]:
            if item["name"] == "x":
                x_shape = item["shape"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "mask":
                    t = torch.rand(item["shape"]) > 0.5
                elif item["name"] == "position":
                    # prefix sum of mask
                    t = torch.cumsum(torch.rand(item["shape"]) > 0.5, dim=-1).to(dtype)
                else:
                    t = torch.randn(item["shape"], dtype=dtype)
                    if dtype == torch.int32:
                        t = (t * 10).clamp(-10000, 10000).to(torch.int32)
                inputs.append(t)
            elif item["type"] == "attr":
                inputs.append(item["value"])
        groups.append(inputs)
    return groups


def get_init_inputs():
    return []
