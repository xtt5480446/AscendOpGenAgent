import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """FeedsRepeat: repeat feeds rows according to repeat times."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        repeat_times = feeds_repeat_times.tolist()
        repeated = []
        for i, t in enumerate(repeat_times):
            repeated.append(feeds[i:i+1].repeat(int(t), 1))
        result = torch.cat(repeated, dim=0)
        if result.shape[0] < output_feeds_size:
            pad = torch.zeros(output_feeds_size - result.shape[0], result.shape[1], dtype=result.dtype, device=result.device)
            result = torch.cat([result, pad], dim=0)
        return result[:output_feeds_size]


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_FeedsRepeat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        feeds_info = inputs[0]
        dtype = dtype_map[feeds_info["dtype"]]
        feeds = torch.randn(feeds_info["shape"], dtype=dtype)
        times_dtype_map = {"int32": torch.int32, "int64": torch.int64}
        times_info = inputs[1]
        times_dtype = times_dtype_map[times_info["dtype"]]
        feeds_repeat_times = torch.tensor(times_info["value"], dtype=times_dtype)
        output_feeds_size = inputs[2]["value"]
        input_groups.append([feeds, feeds_repeat_times, output_feeds_size])
    return input_groups


def get_init_inputs():
    return []
