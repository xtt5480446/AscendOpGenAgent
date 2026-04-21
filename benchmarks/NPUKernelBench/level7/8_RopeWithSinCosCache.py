import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs RopeWithSinCosCache.
    Uses cached sin/cos values via positions indexing.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, positions: torch.Tensor, queryIn: torch.Tensor, keyIn: torch.Tensor,
                cosSinCache: torch.Tensor, mropeSection: torch.Tensor, headSize: int,
                isNeoxStyle: bool):
        """
        Args:
            positions: Index tensor, [numTokens] for rope or [3, numTokens] for mrope.
            queryIn: Query tensor, 2D [numTokens, headSize].
            keyIn: Key tensor, 2D [numTokens, headSize].
            cosSinCache: Cache tensor, [maxSeqLen, 2*headSize].
            mropeSection: Section sizes for mrope, [3].
            headSize: Head dimension size.
            isNeoxStyle: True for rotate_half, False for rotate_interleaved.
        Returns:
            queryOut, keyOut.
        """
        # Index cache
        cosSin = cosSinCache[positions]  # [numTokens, 2*headSize] or [3, numTokens, 2*headSize]
        cos, sin = cosSin.chunk(2, dim=-1)

        if positions.dim() == 2 and positions.shape[0] == 3:
            # mrope mode
            cos0 = cos[0, :, :mropeSection[0]]
            cos1 = cos[1, :, mropeSection[0]:mropeSection[0]+mropeSection[1]]
            cos2 = cos[2, :, mropeSection[0]+mropeSection[1]:mropeSection[0]+mropeSection[1]+mropeSection[2]]
            cos = torch.cat((cos0, cos1, cos2), dim=-1)
            sin0 = sin[0, :, :mropeSection[0]]
            sin1 = sin[1, :, mropeSection[0]:mropeSection[0]+mropeSection[1]]
            sin2 = sin[2, :, mropeSection[0]+mropeSection[1]:mropeSection[0]+mropeSection[1]+mropeSection[2]]
            sin = torch.cat((sin0, sin1, sin2), dim=-1)

        rotaryDim = cos.shape[-1]
        queryRot = queryIn[..., :rotaryDim]
        queryPass = queryIn[..., rotaryDim:]
        keyRot = keyIn[..., :rotaryDim]
        keyPass = keyIn[..., rotaryDim:]

        if isNeoxStyle:
            x1, x2 = queryRot.chunk(2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            queryRot = torch.cat((o1, o2), dim=-1)
        else:
            x1 = queryRot[..., ::2]
            x2 = queryRot[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            queryRot = torch.stack((o1, o2), dim=-1).flatten(-2)

        queryOut = torch.cat((queryRot, queryPass), dim=-1)

        if isNeoxStyle:
            x1, x2 = keyRot.chunk(2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            keyRot = torch.cat((o1, o2), dim=-1)
        else:
            x1 = keyRot[..., ::2]
            x2 = keyRot[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            keyRot = torch.stack((o1, o2), dim=-1).flatten(-2)

        keyOut = torch.cat((keyRot, keyPass), dim=-1)
        return queryOut, keyOut


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_RopeWithSinCosCache.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
    }

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        group = []
        for inp in inputs:
            if inp["type"] == "tensor":
                dtype = dtype_map[inp["dtype"]]
                if inp["dtype"] in ("float32", "float16", "bfloat16"):
                    if "cosSin" in inp["name"]:
                        t = torch.randn(inp["shape"], dtype=torch.float32)
                        group.append(torch.cat([torch.cos(t[..., :t.shape[-1]//2]), torch.sin(t[..., t.shape[-1]//2:])], dim=-1).to(dtype))
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
