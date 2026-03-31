# Baseline

> main @ latest, 2026-03-28
> Device: Ascend 910B2, CANN 8.0, PyTorch 2.1
> 评测范围: Level 1 (34 tasks), Level 2 (6 tasks), 共 40 tasks

## KernelBench

### 评测子集列表：

Level 1：2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 57, 61, 63, 64, 67, 82, 87, 99, 100

Level 2：6, 12, 17, 23, 30, 94
### 基线结果（更新于2026-03-28）

| Level | Problem ID | Problem |    编译   |    精度  | PyTorch (ms) | Generated (ms) | Speedup |
|:---|:---:|---------|:-------:|:------:|-------------:|---------------:|--------:|
| 1 | 2 | Standard matrix multiplication |   Pass  |   Pass |         1.65 |           1.70 |   0.97x |
| 1 | 4 | Matrix vector multiplication |   Pass  |   Pass |        36.94 |           9.54 |   3.87x |
| 1 | 10 | 3D tensor matrix multiplication |   Pass  |   Pass |         0.62 |           0.81 |   0.76x |
| 1 | 11 | 4D tensor matrix multiplication |   Pass  |   Pass |         5.67 |           6.08 |   0.93x |
| 1 | 12 | Matmul with diagonal matrices |   Pass  |   Pass |         0.07 |           0.09 |   0.75x |
| 1 | 13 | Matmul for symmetric matrices |   Pass  |   Pass |         1.65 |           1.68 |   0.98x |
| 1 | 14 | Matmul for upper triangular matrices |   Pass  |   Pass |         1.66 |          12.78 |   0.13x |
| 1 | 15 | Matmul for lower triangular matrices |   Pass  |   Pass |         1.71 |           1.76 |   0.98x |
| 1 | 16 | Matmul with transposed A |   Pass  |   Pass |         1.65 |        1054.70 |   0.00x |
| 1 | 17 | Matmul with transposed B |  Pass |  Pass |         1.65 |        4983.07 |   0.00x |
| 1 | 33 | BatchNorm |   Pass  |   Pass |        10.09 |        2653.64 |   0.00x |
| 1 | 34 | InstanceNorm |  Pass |  Pass |        16.68 |         128.41 |   0.13x |
| 1 | 35 | GroupNorm |  Pass |  Pass |        17.96 |          93.06 |   0.19x |
| 1 | 36 | RMSNorm |  Pass |  Pass |        33.60 |        1146.41 |   0.03x |
| 1 | 41 | Max_Pooling_1D |  Pass |  Pass |        22.64 |        6583.55 |  0.003x |
| 1 | 42 | Max_Pooling_2D |   Pass  |   Pass |            - |              - |       - |
| 1 | 43 | Max_Pooling_3D |   Pass  |   Pass |            - |           8.34 |       - |
| 1 | 44 | Average_Pooling_1D |   Pass  |   Pass |        20.52 |          63.92 |   0.32x |
| 1 | 45 | Average_Pooling_2D |  Pass |  Pass |         3.89 |           2.67 |   1.46x |
| 1 | 46 | Average_Pooling_3D |  Pass |  Pass |         0.17 |           0.08 |    2.06 |
| 1 | 48 | Mean_reduction_over_a_dimension |   Pass  |   Pass |         7.69 |         248.97 |   0.03x |
| 1 | 50 | conv_standard_2D_square_input_square_kernel |   Pass  |   Pass |            - |              - |       - |
| 1 | 51 | Argmax_over_a_dimension |   Pass  |   Pass |            - |              - |       - |
| 1 | 53 | Min_reduction_over_a_dimension |   Pass  |   Pass |        18.13 |          22.97 |   0.79x |
| 1 | 54 | conv_standard_3D_square_input_square_kernel |   Pass  |   Pass |            - |              - |       - |
| 1 | 57 | conv_transposed_2D_square_input_square_kernel |  Failed |  Failed |            - |              - |   1.00x |
| 1 | 61 | conv_transposed_3D_square_input_square_kernel |   Pass  |   Pass |            - |              - |   0.86x |
| 1 | 63 | conv_standard_2D_square_input_square_kernel |  Failed |  Failed |        23.59 |         166.72 |   0.14x |
| 1 | 64 | conv_transposed_1D |  Failed |  Failed |            - |              - |   1.00x |
| 1 | 67 | conv_standard_1D |   Pass  |   Pass |            - |              - |       - |
| 1 | 82 | conv_depthwise_2D_square_input_square_kernel |   Pass  |   Pass |            - |              - |   0.12x |
| 1 | 87 | conv_pointwise_2D |   Pass  |   Pass |        31.63 |        4304.04 |   0.01x |
| 1 | 99 | TripletMarginLoss |   Pass  |   Pass |        10.68 |           4.60 |   2.32x |
| 1 | 100 | HingeLoss |   Pass  |   Pass |        31.67 |        1283.66 |   0.02x |
| 2 | 6 | Conv3d_Softmax_MaxPool_MaxPool |   Pass  |   Pass |         0.48 |              - |   0.00x |
| 2 | 12 | Gemm_Multiply_LeakyReLU |   Pass  |   Pass |         0.61 |           0.62 |   0.98x |
| 2 | 17 | Conv2d_InstanceNorm_Divide |   Pass  |   Pass |         3.03 |           3.25 |   0.93x |
| 2 | 23 | Conv3d_GroupNorm_Mean |   Pass  |   Pass |         0.65 |           0.65 |   1.00x |
| 2 | 30 | Gemm_GroupNorm_Hardtanh |   Pass  |   Pass |         0.66 |           0.63 |   1.04x |
| 2 | 94 | Gemm_BiasAdd_Hardtanh_Mish_GroupNorm | Failed | Failed |         0.66 |           0.66 |   1.00x |

编译通过率：36/40
精度通过率：36/40
综合通过率（编译+精度均过）：36/40
平均 Speedup：0.70x（仅统计有性能数据的 31 个任务）
