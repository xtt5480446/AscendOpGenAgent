# WholeReduceMin-归约计算-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0080
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0080.html
---

# WholeReduceMin

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

每个repeat内所有数据求最小值以及其索引index，返回的索引值为每个repeat内部索引。归约指令的总体介绍请参考如何使用归约计算API。

#### 函数原型

- mask逐bit模式：12template<typenameT,boolisSetMask=true>__aicore__inlinevoidWholeReduceMin(constLocalTensor<T>&dst,constLocalTensor<T>&src,constuint64_tmask[],constint32_trepeatTime,constint32_tdstRepStride,constint32_tsrcBlkStride,constint32_tsrcRepStride,ReduceOrderorder=ReduceOrder::ORDER_VALUE_INDEX)
- mask连续模式：12template<typenameT,boolisSetMask=true>__aicore__inlinevoidWholeReduceMin(constLocalTensor<T>&dst,constLocalTensor<T>&src,constint32_tmask,constint32_trepeatTime,constint32_tdstRepStride,constint32_tsrcBlkStride,constint32_tsrcRepStride,ReduceOrderorder=ReduceOrder::ORDER_VALUE_INDEX)

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/floatAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/floatAtlas 200I/500 A2 推理产品，支持的数据类型为：half/floatAtlas 推理系列产品AI Core，支持的数据类型为：half/floatAtlas 训练系列产品，支持的数据类型为：half |
| isSetMask | 是否在接口内部设置mask。true，表示在接口内部设置mask。false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证4字节对齐（针对half数据类型），8字节对齐（针对float数据类型）。 |
| src | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。源操作数的数据类型需要与目的操作数保持一致。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 迭代次数。取值范围为[0, 255]。关于该参数的具体描述请参考高维切分API。 |
| dstRepStride | 输入 | 目的操作数相邻迭代间的地址步长。以一个repeat归约后的长度为单位。返回索引和最值时，单位为dst数据类型所占字节长度的两倍。比如当dst为half时，单位为4Bytes；仅返回最值时，单位为dst数据类型所占字节长度；仅返回索引时，单位为uint32_t类型所占字节长度。注意，此参数值Atlas 训练系列产品不支持配置0。 |
| srcBlkStride | 输入 | 单次迭代内datablock的地址步长。详细说明请参考dataBlockStride。 |
| srcRepStride | 输入 | 源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考repeatStride。 |
| order | 输入 | 使用order参数指定dst中index与value的相对位置以及返回结果行为，ReduceOrder类型，默认值为ORDER_VALUE_INDEX。取值范围如下：ORDER_VALUE_INDEX：表示value位于低半部，返回结果存储顺序为[value, index]。ORDER_INDEX_VALUE：表示index位于低半部，返回结果存储顺序为[index, value]。ORDER_ONLY_VALUE：表示只返回最值，返回结果存储顺序为[value]。ORDER_ONLY_INDEX：表示只返回最值索引，返回结果存储顺序为[index]。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。Atlas 200I/500 A2 推理产品，支持ORDER_VALUE_INDEX、ORDER_ONLY_VALUE。Atlas 推理系列产品AI Core，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE。Atlas 训练系列产品，支持ORDER_VALUE_INDEX。 |

#### 返回值说明

无

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 操作数地址重叠约束请参考通用地址重叠约束。

- dst结果存储顺序由order决定，默认为最值，最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。比如调用示例中，前两个计算结果为[9.980e-01 5.364e-06]，5.364e-06需要使用reinterpret_cast方法转换得到索引值90。特别地，针对Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品，ORDER_ONLY_INDEX（仅返回最值索引）情况下，读取index时都需要使用reinterpret_cast<uint32_t*>。
- 针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。

#### 调用示例

| 12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455 | #include"kernel_operator.h"classKernelReduce{public:__aicore__inlineKernelReduce(){}__aicore__inlinevoidInit(__gm__uint8_t*src,__gm__uint8_t*dstGm){srcGlobal.SetGlobalBuffer((__gm__half*)src);dstGlobal.SetGlobalBuffer((__gm__half*)dstGm);repeat=srcDataSize/mask;pipe.InitBuffer(inQueueSrc,1,srcDataSize*sizeof(half));pipe.InitBuffer(outQueueDst,1,dstDataSize*sizeof(half));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.AllocTensor<half>();AscendC::DataCopy(srcLocal,srcGlobal,srcDataSize);inQueueSrc.EnQue(srcLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.DeQue<half>();AscendC::LocalTensor<half>dstLocal=outQueueDst.AllocTensor<half>();AscendC::WholeReduceMin<half>(dstLocal,srcLocal,mask,repeat,1,1,8);// 使用默认order, ReduceOrder::ORDER_VALUE_INDEXoutQueueDst.EnQue<half>(dstLocal);inQueueSrc.FreeTensor(srcLocal);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<half>dstLocal=outQueueDst.DeQue<half>();AscendC::DataCopy(dstGlobal,dstLocal,dstDataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<half>srcGlobal,dstGlobal;intsrcDataSize=1024;intdstDataSize=16;intmask=128;intrepeat=0;};extern"C"__global____aicore__voidreduce_kernel(__gm__uint8_t*src,__gm__uint8_t*dstGm){KernelReduceop;op.Init(src,dstGm);op.Process();} |
| --- | --- |

示例结果如下：
