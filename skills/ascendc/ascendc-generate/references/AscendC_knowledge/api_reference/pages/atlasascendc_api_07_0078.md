# ReduceSum-归约计算-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0078
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0078.html
---

# ReduceSum

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

对所有的输入数据求和。

- 方式一：同一repeat内先按照二叉树累加、不同repeat的结果也按照二叉树累加。假设源操作数为128个half类型的数据[data0,data1,data2...data127]，一个repeat可以计算完，计算过程如下。data0和data1相加得到data00，data2和data3相加得到data01，...，data124和data125相加得到data62，data126和data127相加得到data63；data00和data01相加得到data000，data02和data03相加得到data001，...，data62和data63相加得到data031；以此类推，得到目的操作数为1个half类型的数据[data]。需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。
- 方式二：同一repeat内采用二叉树累加，不同repeat的结果按顺序累加。

不同硬件形态对应的ReduceSum相加方式如下：

Atlas A3 训练系列产品/Atlas A3 推理系列产品tensor前n个数据计算接口采用方式二，tensor高维切分计算接口采用方式一

Atlas A2 训练系列产品/Atlas A2 推理系列产品tensor前n个数据计算接口采用方式二，tensor高维切分计算接口采用方式一

Atlas 200I/500 A2 推理产品采用方式一

Atlas 推理系列产品AI Core采用方式一

Atlas 训练系列产品采用方式一

- sharedTmpBuffer支持两种处理方式：方式一：按照如下计算公式计算最小所需空间：12345678910111213141516171819// 先定义一个向上取整函数intRoundUp(inta,intb){return(a+b-1)/b;}// 然后定义参与计算的数据类型inttypeSize=2;// half类型为2Bytes，float类型为4Bytes，按需填入// 再根据数据类型定义两个单位intelementsPerBlock=32/typeSize;// 1个datablock存放的元素个数intelementsPerRepeat=256/typeSize;// 1次repeat可以处理的元素个数// 最后确定首次最大repeat值intfirstMaxRepeat=repeatTime;// 此处需要注意：对于tensor高维切分计算接口，firstMaxRepeat就是repeatTime；对于tensor前n个数据计算接口，firstMaxRepeat为count/elementsPerRepeat，比如在half类型下firstMaxRepeat就是count/128，在float类型下为count/64，按需填入，对于count<elementsPerRepeat的场景，firstMaxRepeat就是1intiter1OutputCount=firstMaxRepeat;// 第一轮操作产生的元素个数intiter1AlignEnd=RoundUp(iter1OutputCount,elementsPerBlock)*elementsPerBlock;// 第一轮产生的元素个数做向上取整intfinalWorkLocalNeedSize=iter1AlignEnd;// 最终sharedTmpBuffer所需的elements空间大小就是第一轮操作产生元素做向上取整后的结果方式二：传入任意大小的sharedTmpBuffer，sharedTmpBuffer的值不会被改变。

#### 函数原型

- tensor前n个数据计算12template<typenameT,boolisSetMask=true>__aicore__inlinevoidReduceSum(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<T>&sharedTmpBuffer,constint32_tcount)
- tensor高维切分计算mask逐bit模式12template<typenameT>__aicore__inlinevoidReduceSum(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<T>&sharedTmpBuffer,constuint64_tmask[],constint32_trepeatTime,constint32_tsrcRepStride)mask连续模式12template<typenameT>__aicore__inlinevoidReduceSum(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<T>&sharedTmpBuffer,constint32_tmask,constint32_trepeatTime,constint32_tsrcRepStride)

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/floatAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/floatAtlas 200I/500 A2 推理产品，支持的数据类型为：half/floatAtlas 推理系列产品AI Core，支持的数据类型为：half/floatAtlas 训练系列产品，支持的数据类型为：half |
| isSetMask | 预留参数，为后续的功能做保留。保持默认值即可。 |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要保证2字节对齐（针对half数据类型），4字节对齐（针对float数据类型）。 |
| src | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 指令执行期间用于存储中间结果，用于内部计算所需操作空间，需特别注意空间大小，参见约束说明。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。 |
| count | 输入 | 参与计算的元素个数。参数取值范围和操作数的数据类型有关，数据类型不同，能够处理的元素个数最大值不同，最大处理的数据量不能超过UB大小限制。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 迭代次数。与通用参数说明中不同的是，支持更大的取值范围，保证不超过int32_t最大值的范围即可。 |
| srcRepStride | 输入 | 源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考repeatStride。 |

#### 返回值说明

无

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 操作数地址重叠约束请参考通用地址重叠约束。需要使用sharedTmpBuffer的情况下，支持dst与sharedTmpBuffer地址重叠（通常情况下dst比sharedTmpBuffer所需的空间要小），此时sharedTmpBuffer必须满足最小所需空间要求，否则不支持地址重叠。

- 该接口内部通过软件仿真来实现ReduceSum功能，某些场景下，性能可能不及直接使用硬件指令实现的BlockReduceSum和WholeReduceSum接口。针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。

#### 调用示例

- tensor高维切分计算样例-mask连续模式123// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算int32_tmask=128;AscendC::ReduceSum<half>(dstLocal,srcLocal,sharedTmpBuffer,mask,65,8);
- tensor高维切分计算样例-mask逐bit模式123// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，使用tensor高维切分计算接口，设定repeatTime为65，mask为全部元素参与计算uint64_tmask[2]={0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF};AscendC::ReduceSum<half>(dstLocal,srcLocal,sharedTmpBuffer,mask,65,8);
- tensor前n个数据计算样例12// dstLocal,srcLocal和sharedTmpBuffer均为half类型,srcLocal的计算数据量为8320,并且连续排布，使用tensor前n个数据计算接口AscendC::ReduceSum<half>(dstLocal,srcLocal,sharedTmpBuffer,8320);

- tensor高维切分计算接口完整示例:12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455#include"kernel_operator.h"classKernelReduce{public:__aicore__inlineKernelReduce(){}__aicore__inlinevoidInit(__gm__uint8_t*src,__gm__uint8_t*dstGm){srcGlobal.SetGlobalBuffer((__gm__half*)src);dstGlobal.SetGlobalBuffer((__gm__half*)dstGm);repeat=srcDataSize/mask;pipe.InitBuffer(inQueueSrc,1,srcDataSize*sizeof(half));pipe.InitBuffer(workQueue,1,80*sizeof(half));// 此处按照公式计算所需的最小work空间为80，也就是160Bytespipe.InitBuffer(outQueueDst,1,dstDataSize*sizeof(half));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.AllocTensor<half>();AscendC::DataCopy(srcLocal,srcGlobal,srcDataSize);inQueueSrc.EnQue(srcLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.DeQue<half>();AscendC::LocalTensor<half>dstLocal=outQueueDst.AllocTensor<half>();AscendC::LocalTensor<half>sharedTmpBuffer=workQueue.AllocTensor<half>();// level0AscendC::ReduceSum<half>(dstLocal,srcLocal,sharedTmpBuffer,mask,repeat,repStride);outQueueDst.EnQue<half>(dstLocal);inQueueSrc.FreeTensor(srcLocal);workQueue.FreeTensor(sharedTmpBuffer);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<half>dstLocal=outQueueDst.DeQue<half>();AscendC::DataCopy(dstGlobal,dstLocal,dstDataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc;AscendC::TQue<AscendC::TPosition::VECOUT,1>workQueue;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<half>srcGlobal,dstGlobal;intsrcDataSize=8320;intdstDataSize=16;intmask=128;intrepStride=8;intrepeat=0;};示例结果如下：输入数据(src_gm):
[1. 1. 1. ... 1. 1. 1.]
输出数据(dst_gm):
[8320.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
    0.    0.    0.    0.]
- tensor前n个数据计算接口完整示例:123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354#include"kernel_operator.h"classKernelReduce{public:__aicore__inlineKernelReduce(){}__aicore__inlinevoidInit(__gm__uint8_t*src,__gm__uint8_t*dstGm){srcGlobal.SetGlobalBuffer((__gm__half*)src);dstGlobal.SetGlobalBuffer((__gm__half*)dstGm);repeat=srcDataSize/mask;pipe.InitBuffer(inQueueSrc,1,srcDataSize*sizeof(half));pipe.InitBuffer(workQueue,1,16*sizeof(half));// 此处按照公式计算所需的最小work空间为16,也就是32Bytespipe.InitBuffer(outQueueDst,1,dstDataSize*sizeof(half));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.AllocTensor<half>();AscendC::DataCopy(srcLocal,srcGlobal,srcDataSize);inQueueSrc.EnQue(srcLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.DeQue<half>();AscendC::LocalTensor<half>dstLocal=outQueueDst.AllocTensor<half>();AscendC::LocalTensor<half>sharedTmpBuffer=workQueue.AllocTensor<half>();AscendC::ReduceSum<half>(dstLocal,srcLocal,sharedTmpBuffer,srcDataSize);outQueueDst.EnQue<half>(dstLocal);inQueueSrc.FreeTensor(srcLocal);workQueue.FreeTensor(sharedTmpBuffer);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<half>dstLocal=outQueueDst.DeQue<half>();AscendC::DataCopy(dstGlobal,dstLocal,dstDataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc;AscendC::TQue<AscendC::TPosition::VECOUT,1>workQueue;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<half>srcGlobal,dstGlobal;intsrcDataSize=288;intdstDataSize=16;intmask=128;intrepStride=8;intrepeat=0;};示例结果如下：输入数据(src_gm):
[1. 1. 1. ... 1. 1. 1.]
输出数据(dst_gm):
[288.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
