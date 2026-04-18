# Gather-离散与聚合-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0091
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0091.html
---

# Gather

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

给定输入的张量和一个地址偏移张量，本接口根据偏移地址将输入张量按元素收集到结果张量中。

#### 函数原型

- tensor前n个数据计算12template<typenameT>__aicore__inlinevoidGather(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<uint32_t>&srcOffset,constuint32_tsrcBaseAddr,constuint32_tcount)
- tensor高维切分计算mask逐bit模式12template<typenameT>__aicore__inlinevoidGather(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<uint32_t>&srcOffset,constuint32_tsrcBaseAddr,constuint64_tmask[],constuint8_trepeatTime,constuint16_tdstRepStride)mask连续模式12template<typenameT>__aicore__inlinevoidGather(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<uint32_t>&srcOffset,constuint32_tsrcBaseAddr,constuint64_tmask,constuint8_trepeatTime,constuint16_tdstRepStride)

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/float/half/bfloat16_tAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/float/half/bfloat16_tAtlas 200I/500 A2 推理产品，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/uint32_t/int32_t/floatAtlas 推理系列产品AI Core，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/float/half |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型和dst保持一致。 |
| srcOffset | 输入 | 每个元素在src中对应的地址偏移。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。该偏移量相对于src的起始基地址而言。单位为Bytes。取值要求如下：取值应保证src元素类型位宽对齐。偏移地址后不能超出UB大小数据的范围。针对以下型号，地址偏移的取值范围：不能超出uint32_t的范围。Atlas 推理系列产品AI CoreAtlas A2 训练系列产品/Atlas A2 推理系列产品Atlas A3 训练系列产品/Atlas A3 推理系列产品针对以下型号，地址偏移的取值范围：当操作数为8位时，取值范围为[0, 216-1]；当操作数为16位时，取值范围为[0, 217-1]，当操作数为32位或者64位时，不能超出uint32_t的范围。Atlas 200I/500 A2 推理产品 |
| srcBaseAddr | 输入 | src的起始基地址，用于指定Gather操作中源操作数的起始位置，单位为Bytes。取值应保证src元素类型位宽对齐，否则会导致非预期行为。 |
| count | 输入 | 执行处理的数据个数。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为8位或16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。参数类型为长度为2的uint64_t类型数组。例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为8位或16位时，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 264-1]；当操作数为64位时，mask[1]为0，mask[0]∈(0, 232-1]。 |
| repeatTime | 输入 | 指令迭代次数，每次迭代完成8个datablock（32Bytes）的数据收集，数据范围：repeatTime∈[0,255]。特别地，针对以下型号：Atlas 200I/500 A2 推理产品操作数为8位时，每次迭代完成4个datablock（32Bytes）的数据收集。 |
| dstRepStride | 输入 | 相邻迭代间的地址步长，单位是datablock（32Bytes）。 |

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 操作数地址重叠约束请参考通用地址重叠约束。

#### 调用示例

| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869 | #include"kernel_operator.h"template<typenameT>classGatherTest{public:__aicore__inlineGatherTest(){}__aicore__inlinevoidInit(__gm__uint8_t*dstGm,__gm__uint8_t*srcGm,__gm__uint8_t*srcOffsetGm,constuint32_tcount){m_elementCount=count;m_dstGlobal.SetGlobalBuffer((__gm__T*)dstGm);m_srcGlobal.SetGlobalBuffer((__gm__T*)srcGm);m_srcOffsetGlobal.SetGlobalBuffer((__gm__uint32_t*)srcOffsetGm);m_pipe.InitBuffer(m_queIn,2,m_elementCount*sizeof(uint32_t));m_pipe.InitBuffer(m_queOut,2,m_elementCount*sizeof(uint32_t));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<T>srcLocal=m_queIn.AllocTensor<T>();AscendC::DataCopy(srcLocal,m_srcGlobal,m_elementCount);m_queIn.EnQue(srcLocal);AscendC::LocalTensor<uint32_t>srcOffsetLocal=m_queIn.AllocTensor<uint32_t>();AscendC::DataCopy(srcOffsetLocal,m_srcOffsetGlobal,m_elementCount);m_queIn.EnQue(srcOffsetLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<T>srcLocal=m_queIn.DeQue<T>();AscendC::LocalTensor<uint32_t>srcOffsetLocal=m_queIn.DeQue<uint32_t>();AscendC::LocalTensor<T>dstLocal=m_queOut.AllocTensor<T>();srcLocal.SetSize(m_elementCount);AscendC::Gather(dstLocal,srcLocal,srcOffsetLocal,(uint32_t)0,m_elementCount);m_queIn.FreeTensor(srcLocal);m_queIn.FreeTensor(srcOffsetLocal);m_queOut.EnQue(dstLocal);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<T>dstLocal=m_queOut.DeQue<T>();AscendC::DataCopy(m_dstGlobal,dstLocal,m_elementCount);m_queOut.FreeTensor(dstLocal);}private:AscendC::TPipem_pipe;AscendC::TQue<AscendC::TPosition::VECIN,1>m_queCalc;AscendC::GlobalTensor<T>m_valueGlobal;uint32_tm_concatRepeatTimes;uint32_tm_sortRepeatTimes;uint32_tm_extractRepeatTimes;uint32_tm_elementCount;AscendC::GlobalTensor<uint32_t>m_srcOffsetGlobal;AscendC::GlobalTensor<T>m_srcGlobal;AscendC::GlobalTensor<T>m_dstGlobal;AscendC::TQue<AscendC::TPosition::VECIN,2>m_queIn;AscendC::TQue<AscendC::TPosition::VECOUT,2>m_queOut;};// class GatherTestextern"C"__global____aicore__voidkernel_gather(GM_ADDRdstGm,GM_ADDRsrcGm,GM_ADDRsrcOffsetGm){GatherTest<half>op;op.Init(dstGm,srcGm,srcOffsetGm,128);op.Process();} |
| --- | --- |
