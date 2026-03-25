# torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
# https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_apply_adam.md

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs AdamW optimization step using NPU accelerated Adam.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                grad: torch.Tensor, beta1_power: float, beta2_power: float,
                lr: float, beta1: float, beta2: float, epsilon: float,
                use_locking: bool = False, use_nesterov: bool = False):
        """
        Applies Adam optimization step on NPU.

        Args:
            var (torch.Tensor): Variable to be updated.
            m (torch.Tensor): First moment estimates.
            v (torch.Tensor): Second moment estimates.
            grad (torch.Tensor): Gradient tensor.
            beta1_power (float): Beta1 power.
            beta2_power (float): Beta2 power.
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment.
            beta2 (float): Exponential decay rate for second moment.
            epsilon (float): Small constant for numerical stability.
            use_locking (bool): Whether to use locking.
            use_nesterov (bool): Whether to use Nesterov momentum.

        Returns:
            tuple: Updated (var, m, v).
        """
        import torch_npu
        torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon,
                                  grad, use_locking, use_nesterov, out=(var, m, v))
        return var, m, v
