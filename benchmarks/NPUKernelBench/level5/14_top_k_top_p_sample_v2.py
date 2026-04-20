import torch
import torch.nn as nn


class Model(nn.Module):
    """
    TopKTopPSampleV2: performs top-k/top-p/minP sampling on logits.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, logits: torch.Tensor, topK: torch.Tensor, topP: torch.Tensor,
                q: torch.Tensor, minPs: torch.Tensor, eps: float, isNeedLogits: int,
                topKGuess: int, ksMAX: int, inputIsLogits: int, isNeedSampleResult: int):
        """
        Args:
            logits (torch.Tensor): [batch, vocab_size].
            topK (torch.Tensor): [batch], k values.
            topP (torch.Tensor): [batch], p values.
            q (torch.Tensor): [batch, vocab_size], sampling weights.
            minPs (torch.Tensor): [batch], minP thresholds.
            eps (float): epsilon for numerical stability.
            isNeedLogits (int): whether to return logits.
            topKGuess (int): candidate size for topP traversal.
            ksMAX (int): max k value.
            inputIsLogits (int): whether input is logits.
            isNeedSampleResult (int): whether to return intermediate results.
        Returns:
            tuple: (logitsSelectIdx, logitsTopKPSelect, logitsIdx, logitsSortMasked)
        """
        batch, vocab = logits.shape
        logits_fp = logits.float()
        logitsSelectIdx = torch.zeros(batch, dtype=torch.int64)
        logitsTopKPSelect = torch.zeros_like(logits_fp)
        logitsIdx = torch.zeros(batch, vocab, dtype=torch.int64)
        logitsSortMasked = torch.zeros_like(logits_fp)
        for b in range(batch):
            logit = logits_fp[b]
            k = topK[b].item()
            p = topP[b].item()
            minP = minPs[b].item()
            # top-k filtering
            if 1 <= k <= min(vocab, ksMAX):
                topk_vals, topk_idx = torch.topk(logit, k)
                mask = logit < topk_vals[-1]
                logit = torch.where(mask, torch.tensor(float('-inf')), logit)
            # softmax
            probs = torch.softmax(logit, dim=-1)
            # minP filtering
            max_prob = probs.max().item()
            minP_threshold = max_prob * minP
            logit = torch.where(probs < minP_threshold, torch.tensor(float('-inf')), logit)
            # cumsum for top-p
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum >= p).nonzero(as_tuple=True)[0]
            if len(cutoff) > 0:
                cutoff_idx = cutoff[0].item()
                threshold = sorted_probs[cutoff_idx]
            else:
                threshold = 0.0
            mask = probs < threshold
            logit = torch.where(mask, torch.tensor(float('-inf')), logit)
            logitsTopKPSelect[b] = logit
            logitsSelectIdx[b] = torch.argmax(logit)
            logitsIdx[b] = sorted_idx
            logitsSortMasked[b] = logit[sorted_idx]
        return logitsSelectIdx, logitsTopKPSelect, logitsIdx, logitsSortMasked


INPUT_CASES = [
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "float16", "shape": [4, 128]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [4]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "float16", "shape": [4]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [4, 128]},
            {"name": "minPs", "type": "tensor", "required": True, "dtype": "float16", "shape": [4]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 32},
            {"name": "ksMAX", "type": "attr", "required": False, "dtype": "int", "value": 1024},
            {"name": "inputIsLogits", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "isNeedSampleResult", "type": "attr", "required": False, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16, 1024]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [16]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [16, 1024]},
            {"name": "minPs", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [16]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 64},
            {"name": "ksMAX", "type": "attr", "required": False, "dtype": "int", "value": 1024},
            {"name": "inputIsLogits", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "isNeedSampleResult", "type": "attr", "required": False, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "float16", "shape": [48, 8192]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 8192]},
            {"name": "minPs", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 128},
            {"name": "ksMAX", "type": "attr", "required": False, "dtype": "int", "value": 1024},
            {"name": "inputIsLogits", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "isNeedSampleResult", "type": "attr", "required": False, "dtype": "int", "value": 0},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "float16", "shape": [48, 131072]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [48]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [48, 131072]},
            {"name": "minPs", "type": "tensor", "required": True, "dtype": "float16", "shape": [48]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 32},
            {"name": "ksMAX", "type": "attr", "required": False, "dtype": "int", "value": 1024},
            {"name": "inputIsLogits", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "isNeedSampleResult", "type": "attr", "required": False, "dtype": "int", "value": 1},
        ]
    },
    {
        "inputs": [
            {"name": "logits", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8, 256]},
            {"name": "topK", "type": "tensor", "required": True, "dtype": "int32", "shape": [8]},
            {"name": "topP", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8]},
            {"name": "q", "type": "tensor", "required": True, "dtype": "float32", "shape": [8, 256]},
            {"name": "minPs", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [8]},
            {"name": "eps", "type": "attr", "required": False, "dtype": "float", "value": 1e-8},
            {"name": "isNeedLogits", "type": "attr", "required": False, "dtype": "int", "value": 0},
            {"name": "topKGuess", "type": "attr", "required": False, "dtype": "int", "value": 16},
            {"name": "ksMAX", "type": "attr", "required": False, "dtype": "int", "value": 256},
            {"name": "inputIsLogits", "type": "attr", "required": False, "dtype": "int", "value": 1},
            {"name": "isNeedSampleResult", "type": "attr", "required": False, "dtype": "int", "value": 0},
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
                    ksMAX = next((i["value"] for i in case["inputs"] if i["name"] == "ksMAX"), 1024)
                    t = torch.randint(1, min(vocab, ksMAX) + 1, (batch,)).to(dtype)
                elif item["name"] in ("topP", "minPs"):
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
