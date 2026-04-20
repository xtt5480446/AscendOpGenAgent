import torch
import torch.nn as nn


class Model(nn.Module):
    """
    TopKTopPSample: performs top-k/top-p sampling on logits.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, logits: torch.Tensor, topK: torch.Tensor, topP: torch.Tensor,
                q: torch.Tensor, eps: float, isNeedLogits: int, topKGuess: int):
        """
        Args:
            logits (torch.Tensor): [batch, vocab_size].
            topK (torch.Tensor): [batch], k values.
            topP (torch.Tensor): [batch], p values.
            q (torch.Tensor): [batch, vocab_size], sampling weights.
            eps (float): epsilon for numerical stability.
            isNeedLogits (int): whether to return logits.
            topKGuess (int): candidate size for topP traversal.
        Returns:
            tuple: (logitsSelectIdx, logitsTopKPSelect)
        """
        batch, vocab = logits.shape
        logits_fp = logits.float()
        logitsSelectIdx = torch.zeros(batch, dtype=torch.int64)
        logitsTopKPSelect = torch.zeros_like(logits_fp)
        for b in range(batch):
            logit = logits_fp[b]
            k = topK[b].item()
            p = topP[b].item()
            # top-k filtering
            if 1 <= k <= min(vocab, 1024):
                topk_vals, topk_idx = torch.topk(logit, k)
                mask = logit < topk_vals[-1]
                logit = torch.where(mask, torch.tensor(float('-inf')), logit)
            # softmax
            probs = torch.softmax(logit, dim=-1)
            # cumsum for top-p
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # find cutoff
            cutoff = (cumsum >= p).nonzero(as_tuple=True)[0]
            if len(cutoff) > 0:
                cutoff_idx = cutoff[0].item()
                threshold = sorted_probs[cutoff_idx]
            else:
                threshold = 0.0
            # mask out values below threshold
            mask = probs < threshold
            logit = torch.where(mask, torch.tensor(float('-inf')), logit)
            logitsTopKPSelect[b] = logit
            # select max idx
            logitsSelectIdx[b] = torch.argmax(logit)
        return logitsSelectIdx, logitsTopKPSelect


INPUT_CASES = [
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "float16", "shape": [4, 128]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "float16", "shape": [4]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 128]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 32},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16, 1024]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 1024]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 64},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "float16", "shape": [48, 8192]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 8192]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 128},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "float16", "shape": [48, 131072]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 131072]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 32},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 256]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 256]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 16},
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
        logits_shape = None
        for item in case["inputs"]:
            if item["name"] == "logits":
                logits_shape = item["shape"]
        for item in case["inputs"]:
            if item["type"] == "tensor":
                dtype = _DTYPE_MAP[item["dtype"]]
                if item["name"] == "topK":
                    batch = item["shape"][0]
                    vocab = logits_shape[1]
                    t = torch.randint(1, min(vocab, 1024) + 1, (batch,)).to(dtype)
                elif item["name"] == "topP":
                    batch = item["shape"][0]
                    t = torch.rand(batch, dtype=torch.float32).clamp(0.1, 0.95)
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
