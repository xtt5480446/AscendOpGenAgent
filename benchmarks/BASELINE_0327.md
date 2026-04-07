# Baseline

## Triton代码生成

### KernelBench 评测子集列表

**Level 1** (34 tasks)：2, 4, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 57, 61, 63, 64, 67, 82, 87, 99, 100

**Level 2** (6 tasks)：6, 12, 17, 23, 30, 94

### Triton-ascend基线结果

**评测环境**
- 分支：main @ latest
- 更新时间：2026-03-27
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.5.1, PyTorch 2.9.0
- 评测范围：Level 1 (34 tasks) + Level 2 (6 tasks) = 40 tasks

**评测综述**

| 指标 | 结果 |
|------|------|
| **精度通过率** | 36/40 (90%) |
| **性能≥0.6x达标** | 16/40 (40%) |
| **性能≥0.8x达标** | 13/40 (32.5%) |
| 平均加速比 | 0.70x（31个有性能数据的任务） |



| Level | Problem ID | Problem | 算子类型 |    精度  | PyTorch (ms) | Triton (ms) | 加速比 | 性能≥0.6x | 性能≥0.8x |
|:---|:---:|---------|:-------:|:------:|-------------:|---------------:|--------:|:-------:|:-------:|
| 1 | 2 | Standard matrix multiplication | CUBE |   ✅ |         1.65 |           1.70 |   0.97x |   ✅   |   ✅   |
| 1 | 4 | Matrix vector multiplication | CUBE |   ✅ |        36.94 |           9.54 |   3.87x |   ✅   |   ✅   |
| 1 | 10 | 3D tensor matrix multiplication | CUBE |   ✅ |         0.62 |           0.81 |   0.76x |   ✅   |   ❌   |
| 1 | 11 | 4D tensor matrix multiplication | CUBE |   ✅ |         5.67 |           6.08 |   0.93x |   ✅   |   ✅   |
| 1 | 12 | Matmul with diagonal matrices | CUBE |   ✅ |         0.07 |           0.09 |   0.75x |   ✅   |   ❌   |
| 1 | 13 | Matmul for symmetric matrices | CUBE |   ✅ |         1.65 |           1.68 |   0.98x |   ✅   |   ✅   |
| 1 | 14 | Matmul for upper triangular matrices | CUBE |   ✅ |         1.66 |          12.78 |   0.13x |   ❌   |   ❌   |
| 1 | 15 | Matmul for lower triangular matrices | CUBE |   ✅ |         1.71 |           1.76 |   0.98x |   ✅   |   ✅   |
| 1 | 16 | Matmul with transposed A | CUBE |   ✅ |         1.65 |        1054.70 |   0.00x |   ❌   |   ❌   |
| 1 | 17 | Matmul with transposed B | CUBE |  ✅ |         1.65 |        4983.07 |   0.00x |   ❌   |   ❌   |
| 1 | 33 | BatchNorm | VECTOR |   ✅ |        10.09 |        2653.64 |   0.00x |   ❌   |   ❌   |
| 1 | 34 | InstanceNorm | VECTOR |  ✅ |        16.68 |         128.41 |   0.13x |   ❌   |   ❌   |
| 1 | 35 | GroupNorm | VECTOR |  ✅ |        17.96 |          93.06 |   0.19x |   ❌   |   ❌   |
| 1 | 36 | RMSNorm | VECTOR |  ✅ |        33.60 |        1146.41 |   0.03x |   ❌   |   ❌   |
| 1 | 41 | Max_Pooling_1D | VECTOR |  ✅ |        22.64 |        6583.55 |  0.003x |   ❌   |   ❌   |
| 1 | 42 | Max_Pooling_2D | VECTOR |   ✅ |            - |              - |       - |   ❌   |   ❌   |
| 1 | 43 | Max_Pooling_3D | VECTOR |   ✅ |            - |           8.34 |       - |   ❌   |   ❌   |
| 1 | 44 | Average_Pooling_1D | VECTOR |   ✅ |        20.52 |          63.92 |   0.32x |   ❌   |   ❌   |
| 1 | 45 | Average_Pooling_2D | VECTOR |  ✅ |         3.89 |           2.67 |   1.46x |   ✅   |   ✅   |
| 1 | 46 | Average_Pooling_3D | VECTOR |  ✅ |         0.17 |           0.08 |    2.06x |   ✅   |   ✅   |
| 1 | 48 | Mean_reduction_over_a_dimension | VECTOR |   ✅ |         7.69 |         248.97 |   0.03x |   ❌   |   ❌   |
| 1 | 50 | conv_standard_2D_square_input_square_kernel | VECTOR |   ✅ |            - |              - |       - |   ❌   |   ❌   |
| 1 | 51 | Argmax_over_a_dimension | VECTOR |   ✅ |            - |              - |       - |   ❌   |   ❌   |
| 1 | 53 | Min_reduction_over_a_dimension | VECTOR |   ✅ |        18.13 |          22.97 |   0.79x |   ✅   |   ❌   |
| 1 | 54 | conv_standard_3D_square_input_square_kernel | CUBE |   ✅ |            - |              - |       - |   ❌   |   ❌   |
| 1 | 57 | conv_transposed_2D_square_input_square_kernel | CUBE |  ❌ |            - |              - |   - |   ❌   |   ❌   |
| 1 | 61 | conv_transposed_3D_square_input_square_kernel | CUBE |   ✅ |            - |              - |   0.86x |   ✅   |   ✅   |
| 1 | 63 | conv_standard_2D_square_input_square_kernel | CUBE |  ❌ |        - |         - |   - |   ❌   |   ❌   |
| 1 | 64 | conv_transposed_1D | CUBE |  ❌ |            - |              - |   - |   ❌   |   ❌   |
| 1 | 67 | conv_standard_1D | CUBE |   ✅ |            - |              - |       - |   -   |   ❌   |
| 1 | 82 | conv_depthwise_2D_square_input_square_kernel | CUBE |   ✅ |            - |              - |   0.12x |   ❌   |   ❌   |
| 1 | 87 | conv_pointwise_2D | CUBE |   ✅ |        31.63 |        4304.04 |   0.01x |   ❌   |   ❌   |
| 1 | 99 | TripletMarginLoss | VECTOR |   ✅ |        10.68 |           4.60 |   2.32x |   ✅   |   ✅   |
| 1 | 100 | HingeLoss | VECTOR |   ✅ |        31.67 |        1283.66 |   0.02x |   ❌   |   ❌   |
| 2 | 6 | Conv3d_Softmax_MaxPool_MaxPool | CV |   ✅ |         0.48 |              - |   0.00x |   ❌   |   ❌   |
| 2 | 12 | Gemm_Multiply_LeakyReLU | CV |   ✅ |         0.61 |           0.62 |   0.98x |   ✅   |   ✅   |
| 2 | 17 | Conv2d_InstanceNorm_Divide | CV |   ✅ |         3.03 |           3.25 |   0.93x |   ✅   |   ✅   |
| 2 | 23 | Conv3d_GroupNorm_Mean | CV |   ✅ |         0.65 |           0.65 |   1.00x |   ✅   |   ✅   |
| 2 | 30 | Gemm_GroupNorm_Hardtanh | CV |   ✅ |         0.66 |           0.63 |   1.04x |   ✅   |   ✅   |
| 2 | 94 | Gemm_BiasAdd_Hardtanh_Mish_GroupNorm | CV | ❌ |         - |           - |   - |   ❌   |   ❌   |




## AscendC代码生成

### NpuKernelBench 评测子集列表

**Level 1** (31 tasks)：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31

**单shape** (45 tasks)

### Ascend C基线结果

**评测环境**
- 分支：main @ latest
- 更新时间：2026-03-27
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.0, PyTorch 2.1
- 评测范围：Level 1 (31 tasks) + 单shape (45 tasks) = 76 tasks（16个算子无描述）

**评测综述**

| 指标 | 结果 |
|------|------|
| **精度通过率** | 16/76 (21%) |
| **性能≥0.6x达标** | 15/76 (19%) |
| **性能≥0.8x达标** | 14/76 (18%) |
| 平均加速比 | 1.52x（15个有性能数据的任务） |


| Level | Problem ID | Problem | 算子类型 |    精度  | PyTorch (ms) | AscendC (ms) | 加速比 | 性能≥0.6x | 性能≥0.8x |
|:---|:---:|---------|:-------:|:------:|-------------:|---------------:|--------:|:-------:|:-------:|
| 1 | 1 | AdaptiveAvgPool3d | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 2 | AdaptiveAvgPool3dGrad | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 3 | AdaptiveMaxPool3d | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 4 | AdaptiveMaxPool3DGrad | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 5 | TransformBiasRescaleQkv | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 6 | AddRmsNormDynamicQuantV2 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 7 | STFT | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 8 | ApplyTopKTopPWithSorted | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 9 | AvgPool3D | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 10 | AvgPool3DGrad | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 11 | BatchNormV3 | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 12 | ChamferDistanceGrad | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 13 | CrossV2 | VECTOR | ✅ | 0.022 | 0.024 | 0.91x | ✅ | ✅ |
| 1 | 14 | CTCLossV3 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 15 | CTCLossV3Grad | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 16 | EmbeddingBag | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 17 | EmbeddingDenseGradV2 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 18 | FatreluMul | VECTOR | ✅ | 0.042ms | 0.027ms | 1.55x | ✅ | ✅ |
| 1 | 19 | ForeachLerpList | VECTOR | ✅ | 0.063ms | 0.058ms | 1.63x | ✅ | ✅ |
| 1 | 20 | ForeachMaximumScalarList | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 21 | ForeachMinimumScalarList | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 22 | ForeachPowList | VECTOR | ✅ | 0.029ms | 0.014ms | 2.13x | ✅ | ✅ |
| 1 | 23 | ForeachPowScalarList | VECTOR | ✅ | 0.0117ms | 0.0195ms | 0.6x | ✅ | ❌ |
| 1 | 24 | GeluMul | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 25 | GemmaRmsNorm | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 26 | Segsum | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 27 | Rfft1D | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 28 | GroupNormGrad | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 29 | GroupNormSilu | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 30 | Pows | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 31 | MulAddn | VECTOR | ✅ | 0.049 | 0.044 | 1.11x | ✅ | ✅ |
| 1 | 32 | Heaviside | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 33 | LinSpace | VECTOR | ✅ | - | - | - | ❌ | ❌ |
| 1 | 34 | InplaceIndexAddWithSorted | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 35 | InplaceScatterAdd | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 36 | KvRmsNormRopeCache | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 37 | LayerNormGradV3 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 38 | LayerNormV4 | VECTOR | ✅ | 0.71 | 0.539 | 1.32x | ✅ | ✅ |
| 1 | 39 | LinearIndexV2 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 40 | Logit | VECTOR | ✅ | 0.022 | 0.031 | 1.38x | ✅ | ✅ |
| 1 | 41 | LogitGrad | VECTOR | ✅ | 0.108 | 0.028 | 3.89x | ✅ | ✅ |
| 1 | 42 | MaskedSoftmaxWithRelPosBias | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 43 | MaxPool3DWithArgmaxV2 | CV | ✅ | 0.0154 ms | 0.0171 ms | 0.9x | ✅ | ✅ |
| 1 | 44 | MaxPool3DGradWithArgmax | CV | ❌ | - | - | - | ❌ | ❌ |
| 1 | 45 | MseLossGradV2 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 46 | HistogramV2 | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 47 | HansEncode | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 48 | QuantizedBatchNorm | CV | ✅ | 0.571 | 0.235 | 2.43x | ✅ | ✅ |
| 1 | 49 | RepeatInterleaveGrad | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 50 | HansDecode | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 51 | DiagV2 | VECTOR | ✅ | 0.023 | 0.023 | 1x | ✅ | ✅ |
| 1 | 52 | RmsNormQuant | VECTOR | ✅ | 1.2283 | 0.6791 | 1.81x | ✅ | ✅ |
| 1 | 53 | ScatterAddWithSorted | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 54 | SquaredRelu | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 55 | AddLayerNormQuant | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 56 | AddRmsNormDynamicQuant | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 57 | Sinkhorn | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 58 | DuaQuantizeAddLayerNorm | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 59 | QuantizeAddLayerNorm | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 60 | GroupedBiasAddGrad | VECTOR | ✅ | 0.215 | 0.021 | 0.87x | ✅ | ✅ |
| 1 | 61 | AngleV2 | VECTOR | ✅ | 0.024 | 0.018 | 1.32x | ✅ | ✅ |
| 1 | 62 | Add | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 63 | Abs | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 64 | GELU | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 65 | SwiGLU | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 66 | Cumsum | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 67 | Histc | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 68 | Sum | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 69 | Sort | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 70 | TopK | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 71 | LayerNorm | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 72 | GroupNorm | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 73 | Permute | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 74 | Cat | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 75 | Split | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 76 | Pad | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 77 | Repeat | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 78 | AdamW | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 79 | Index | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 80 | IndexPut | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 81 | Gather | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 82 | Scatter | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 83 | Nonzero | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 84 | RepeatInterleave | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 85 | EmbeddingDenseBackward | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 86 | NLLLoss | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 87 | MaxPool3D | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 88 | Interpolate | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 89 | DynamicQuant | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 90 | NMS | VECTOR | ❌ | - | - | - | ❌ | ❌ |
| 1 | 91 | IOU | VECTOR | ❌ | - | - | - | ❌ | ❌ |