# Baseline

## Triton代码生成

### KernelBench 评测子集列表

**所有Vector任务** (48个)：19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100

### Triton-ascend基线结果

**评测环境**
- 分支：br_claudecode @ latest
- 更新时间：2026-04-07
- 硬件：Atlas A2 服务器
- 软件栈：CANN 8.5.1, PyTorch 2.9.0
- 评测范围：所有Vector任务 (48个)

**评测综述**

| 指标 | 结果 |
|------|------|
| **精度通过率** | 48/48 (100%) |
| **性能≥0.6x达标** | 33/48 (68.75%) |
| **性能≥0.8x达标** | 33/48 (68.75%) |
| 平均加速比 | 6.14x ( 注：存在最大加速比 222.29x ) |

**详细结果表**

| Level | Problem ID | 算子名称 | 算子类型 | 验收类型 | 评测子集 | PyTorch 性能(ms) | triton性能(ms) | triton优化后性能(ms) | 加速比 | 加速比(性能优化后) | 精度正确 | 性能 0.6x 达标| 性能0.8x 达标 |
|:---|:---:|---------|:-------:|:------:|:------:|-------------:|---------------:|--------:|--------:|--------:|:-------:|:-------:|:-------:|
| 1 | 19 | 19_ReLU.py | Relu | VECTOR | | 9.33 | 11.96 | | 0.78x | 0.99x | ✅ | ✅ | ✅ |
| 1 | 20 | 20_LeakyReLU.py | Elementwise | VECTOR | | 9.37 | 15.66 | | 0.6x | 0.8x | ✅ | ✅ | ✅ |
| 1 | 21 | 21_Sigmoid.py | Elementwise | VECTOR | | 9.36 | 10.47 | | 0.89x | - | ✅ | ✅ | ✅ |
| 1 | 22 | 22_Tanh.py | Elementwise | VECTOR | | 13.1341 | 12.6526 | | 1.04x | - | ✅ | ✅ | ✅ |
| 1 | 23 | 23_Softmax.py | Reduce & Norm | VECTOR | | 15.3363 | 17.2224 | | 0.89x | 1.23x | ✅ | ✅ | ✅ |
| 1 | 24 | 24_LogSoftmax.py | Reduce & Norm | VECTOR | | 14.9328 | 11.1385 | | 1.34x | 1.46x | ✅ | ✅ | ✅ |
| 1 | 25 | 25_Swish.py | Elementwise | VECTOR | | 4.3112 | 17.9197 | | 1.36x | 1.9x | ✅ | ✅ | ✅ |
| 1 | 26 | 26_GELU_.py | Elementwise | VECTOR | | 9.3711 | 52.0375 | | 0.18x | 0.81x | ✅ | ✅ | ✅ |
| 1 | 27 | 27_SELU_.py | Elementwise | VECTOR | | 9.3456 | 13.2669 | 10.9506 | 0.70x | 0.85x | ✅ | ✅ | ✅ |
| 1 | 28 | 28_HardSigmoid.py | Elementwise | VECTOR | | 9.3049 | 10.1086 | 10.219 | 0.92x | 0.91x | ✅ | ✅ | ✅ |
| 1 | 29 | 29_Softplus.py | Elementwise | VECTOR | | 23.4335 | 14.6634 | 14.8372 | 1.6x | 1.58x | ✅ | ✅ | ✅ |
| 1 | 30 | 30_Softsign.py | Elementwise | VECTOR | | 34.0657 | 9.8571 | 9.5056 | 3.46x | 3.58x | ✅ | ✅ | ✅ |
| 1 | 31 | 31_ELU.py | Elementwise | VECTOR | | 9.3463 | 12.5098 | 10.7179 | 0.75x | 0.87x | ✅ | ✅ | ✅ |
| 1 | 32 | 32_HardTanh.py | Elementwise | VECTOR | | 21.7796 | 13.485 | 10.0745 | 1.62x | 2.21x | ✅ | ✅ | ✅ |
| 1 | 33 | 33_BatchNorm.py | Reduce & Norm | VECTOR | Y | 9.0215 | 72.356 | 72.356 | 0.12x | 0.12x | ✅ | ❌ | ❌ |
| 1 | 34 | 34_InstanceNorm.py | Reduce & Norm | VECTOR | Y | 15.4106 | 33.5952 | 11.0224 | 0.46x | 1.4x | ✅ | ✅ | ✅ |
| 1 | 35 | 35_GroupNorm_.py | Reduce & Norm | VECTOR | Y | 17.97 | 1963.11 | 1355.69 | 0.01x | 0.01x | ✅ | ❌ | ❌ |
| 1 | 36 | 36_RMSNorm_.py | Reduce & Norm | VECTOR | Y | 33.57 | 263.92 | / | 0.13x | / | ✅ | ❌ | ❌ |
| 1 | 37 | 37_FrobeniusNorm_.py | Reduce & Norm | VECTOR | | 15.2562 | 15.7896 | / | 0.97 | / | ✅ | ✅ | ✅ |
| 1 | 38 | 38_L1Norm_.py | Reduce & Norm | VECTOR | | 23.76 | 15.07 | 11.8 | 1.28 | 2.01 | ✅ | ✅ | ✅ |
| 1 | 39 | 39_L2Norm_.py | Reduce & Norm | VECTOR | | 15.38 | 16.83 | 11.84 | 1.3 | 1.42 | ✅ | ✅ | ✅ |
| 1 | 40 | 40_LayerNorm.py | Reduce & Norm | VECTOR | | 2.49 | 1.1 | / | 2.27 | / | ✅ | ✅ | ✅ |
| 1 | 41 | 41_Max_Pooling_1D.py | Reduce & Norm | VECTOR | Y | 213.88 | 2103.58 | 2103.52 | 0.1 | 0.1 | ✅ | ❌ | ❌ |
| 1 | 42 | 42_Max_Pooling_2D.py | Reduce & Norm | VECTOR | Y | 27.9157 | 14546.9719 | / | 0 | / | ✅ | ❌ | ❌ |
| 1 | 43 | 43_Max_Pooling_3D.py | Reduce & Norm | VECTOR | Y | | 6163.08 | 3923.65 | | 1.57 | ✅ | ✅ | ✅ |
| 1 | 44 | 44_Average_Pooling_1D.py | Reduce & Norm | VECTOR | Y | 18.32 | 1256.22 | 1257.92 | 0.01 | 0.01 | ✅ | ❌ | ❌ |
| 1 | 45 | 45_Average_Pooling_2D.py | Reduce & Norm | VECTOR | Y | 3449.12 | 448.11 | 309.36 | 11.15 | 1.45 | ✅ | ✅ | ✅ |
| 1 | 46 | 46_Average_Pooling_3D.py | Reduce & Norm | VECTOR | Y | 153.75 | 11133.95 | 6910.94 | 0.02 | 1.61 | ✅ | ✅ | ✅ |
| 1 | 47 | 47_Sum_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | | 6.34 | 514.5 | 452.96 | 0.01 | 1.14 | ✅ | ✅ | ✅ |
| 1 | 48 | 48_Mean_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 7.57 | 182.03 | / | 0.04 | / | ✅ | ❌ | ❌ |
| 1 | 49 | 49_Max_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | | 16.26 | 396.94 | 364.42 | 0.04 | 1.09 | ✅ | ✅ | ✅ |
| 1 | 50 | 50_conv_standard_2D__square_input__square_kernel | Reduce & Norm | VECTOR | Y | 2.5786 | 0.0116 | / | 222.29 | / | ✅ | ✅ | ✅ |
| 1 | 51 | 51_Argmax_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 18.08 | 9358.42 | 4442.38 | 0.00x | 0.00x | ✅ | ❌ | ❌ |
| 1 | 52 | 52_Argmin_over_a_dimension.py | Reduce & Norm | VECTOR | | 63.6204 | 5353.3452 | / | 0.01x | / | ✅ | ❌ | ❌ |
| 1 | 53 | 53_Min_reduction_over_a_dimension.py | Reduce & Norm | VECTOR | Y | 18.0783 | 5322.2028 | / | 0.00x | / | ✅ | ❌ | ❌ |
| 1 | 88 | 88_MinGPTNewGelu.py | Elementwise | VECTOR | | 4.1719 | 0.7925 | / | 5.26x | / | ✅ | ✅ | ✅ |
| 1 | 89 | 89_cumsum.py | Scan & Loss | VECTOR | | 70.3172 | 2778.5388 | / | 0.03x | / | ✅ | ❌ | ❌ |
| 1 | 90 | 90_cumprod.py | Scan & Loss | VECTOR | | 17839.7 | 2778.75 | 2779.82 | 6.42x | 6.42x | ✅ | ✅ | ✅ |
| 1 | 91 | 91_cumsum_reverse.py | Scan & Loss | VECTOR | | 1331.1 | 2779.74 | 2780.76 | 0.48x | 0.48x | ✅ | ❌ | ❌ |
| 1 | 92 | 92_cumsum_exclusive.py | Scan & Loss | VECTOR | | 117.49 | 2778.63 | 2778.72 | 0.04x | 0.04x | ✅ | ❌ | ❌ |
| 1 | 93 | 93_masked_cumsum.py | Scan & Loss | VECTOR | | 84.2 | 331.6 | 327.46 | 0.25x | 0.26x | ✅ | ❌ | ❌ |
| 1 | 94 | 94_MSELoss | Scan & Loss | VECTOR | | 20.93 | 13.57 | 9.35 | 1.54x | 2.24x | ✅ | ✅ | ✅ |
| 1 | 95 | 95_CrossEntropyLoss | Scan & Loss | VECTOR | | 1.23 | 2.92 | 2.48 | 0.42x | 0.49x | ✅ | ❌ | ❌ |
| 1 | 96 | 96_HuberLoss | Scan & Loss | VECTOR | | 14.1 | 19.48 | 11.61 | 0.72x | 1.21x | ✅ | ✅ | ✅ |
| 1 | 97 | 97_CosineSimilarityLoss.py | Scan & Loss | VECTOR | | 31.65 | 20.89 | 19.59 | 1.51x | 1.62x | ✅ | ✅ | ✅ |
| 1 | 98 | 98_KLDivLoss.py | Scan & Loss | VECTOR | | 7.05 | 1.5 | / | 4.69x | / | ✅ | ✅ | ✅ |
| 1 | 99 | 99_TripletMarginLoss.py | Scan & Loss | VECTOR | Y | 10.8 | 4.46 | 3.09 | 2.40x | 3.49x | ✅ | ✅ | ✅ |
| 1 | 100 | 100_HingeLoss.py | Scan & Loss | VECTOR | Y | 33.96 | 3965.15 | 14.52 | 0.01x | 2.17x | ✅ | ✅ | ✅ |
