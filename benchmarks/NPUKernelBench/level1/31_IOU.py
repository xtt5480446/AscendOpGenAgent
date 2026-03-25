# torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor
# https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_iou.md

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that computes IoU (Intersection over Union) on NPU.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0) -> torch.Tensor:
        """
        Computes IoU between bounding boxes.

        Args:
            bboxes (torch.Tensor): First set of bounding boxes.
            gtboxes (torch.Tensor): Second set of bounding boxes (ground truth).
            mode (int, optional): IoU computation mode (0: IoU, 1: IoF).

        Returns:
            torch.Tensor: IoU values between bounding boxes.
        """
        import torch_npu
        return torch_npu.npu_iou(bboxes, gtboxes, mode=mode)
