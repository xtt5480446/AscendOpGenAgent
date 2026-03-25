# torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)
# https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_nms_v4.md

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Non-Maximum Suppression (NMS) on NPU.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        """
        Performs Non-Maximum Suppression (NMS) on bounding boxes.

        Args:
            boxes (torch.Tensor): Bounding boxes tensor of shape (N, 4).
            scores (torch.Tensor): Scores tensor of shape (N,).
            max_output_size (int): Maximum number of output boxes.
            iou_threshold (float): IoU threshold for suppression.
            scores_threshold (float): Score threshold to filter boxes.
            pad_to_max_output_size (bool, optional): Pad output to max_output_size.

        Returns:
            tuple: (selected_boxes_indices, num_selected_boxes)
        """
        import torch_npu
        # Convert float thresholds to scalar tensors on the same device as input
        iou_threshold_tensor = torch.tensor(iou_threshold, dtype=torch.float32, device=boxes.device)
        scores_threshold_tensor = torch.tensor(scores_threshold, dtype=torch.float32, device=boxes.device)
        return torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold_tensor,
                                     scores_threshold_tensor, pad_to_max_output_size=pad_to_max_output_size)
