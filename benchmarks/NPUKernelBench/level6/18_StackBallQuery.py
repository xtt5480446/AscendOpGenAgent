import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """StackBallQuery: find points within a radius for each center point."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, xyz: torch.Tensor, center_xyz: torch.Tensor, max_radius: float, sample_num: int) -> torch.Tensor:
        batch_size = center_xyz.shape[0]
        idx = torch.zeros(batch_size, center_xyz.shape[1], sample_num, dtype=torch.int32)
        for b in range(batch_size):
            for i in range(center_xyz.shape[1]):
                center = center_xyz[b, i]
                dists = torch.norm(xyz[b] - center.unsqueeze(0), dim=1)
                mask = dists < max_radius
                valid_indices = torch.where(mask)[0]
                if valid_indices.numel() >= sample_num:
                    idx[b, i] = valid_indices[:sample_num].to(torch.int32)
                else:
                    n = valid_indices.numel()
                    if n > 0:
                        idx[b, i, :n] = valid_indices[:n].to(torch.int32)
        return idx


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "18_StackBallQuery.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
        }
        xyz_info = inputs[0]
        dtype = dtype_map[xyz_info["dtype"]]
        xyz = torch.randn(xyz_info["shape"], dtype=dtype)
        center_xyz = torch.randn(inputs[1]["shape"], dtype=dtype)
        max_radius = inputs[2]["value"]
        sample_num = inputs[3]["value"]
        input_groups.append([xyz, center_xyz, max_radius, sample_num])
    return input_groups


def get_init_inputs():
    return []
