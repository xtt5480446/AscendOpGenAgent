# 昇腾社区官网-昇腾万里 让智能无所不及
**页面ID:** atlasascendc_api_07_0581
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0581.html
---

# Round

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

将输入的元素四舍五入到最接近的整数，计算公式如下：

![](../images/atlasascendc_api_07_0581_img_001.png)

举例如下：

Round(3.5) = 4.0

Round(-3.4) = -3.0

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间12template<typenameT,boolisReuseSource=false>__aicore__inlinevoidRound(constLocalTensor<T>&dstTensor,constLocalTensor<T>&srcTensor,constLocalTensor<uint8_t>&sharedTmpBuffer,constuint32_tcalCount)

- 接口框架申请临时空间12template<typenameT,boolisReuseSource=false>__aicore__inlinevoidRound(constLocalTensor<T>&dstTensor,constLocalTensor<T>&srcTensor,constuint32_tcalCount)

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者通过sharedTmpBuffer入参传入和接口框架申请两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetRoundMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。用于Round内部复杂计算时存储中间变量，由开发者提供。临时空间大小BufferSize的获取方式请参考GetRoundMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 返回值说明

无

#### 约束说明

- 只支持四舍五入到整数，不支持四舍五入到小数。如果该数恰好在两个整数中间，即小数部分为0.5，则“向偶数舍入”。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
- 操作数地址对齐要求请参见通用地址对齐约束。

#### 调用示例

完整的调用样例请参考更多样例。

| 123456 | AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECCALC,1>tmpQue;pipe.InitBuffer(tmpQue,1,bufferSize);// bufferSize 通过Host侧tiling参数获取AscendC::LocalTensor<uint8_t>sharedTmpBuffer=tmpQue.AllocTensor<uint8_t>();// 输入tensor长度为1024, 算子输入的数据类型为half, 实际计算个数为512AscendC::Round(dstLocal,srcLocal,sharedTmpBuffer,512); |
| --- | --- |

| 12 | 输入数据(srcLocal):[0.805411340.83857050.49426016...1.309622050.28947052]输出数据(dstLocal):[1.01.00.0...1.00.0] |
| --- | --- |
