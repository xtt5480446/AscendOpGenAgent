# Select-比较与选择-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0070
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0070.html
---

# Select

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

给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时从src1选取。

对于tensor高维切分计算接口，支持根据mask参数对上述选取结果，再次进行过滤，有效位填入最终的dst，无效位则保持dst原始值。例如：src0为[1,2,3,4,5,6,7,8]，src1为[9,10,11,12,13,14,15,16]，selMask为[0,0,0,0,1,1,1,1]，mask为[1,1,1,1,0,0,0,0]，dst原始值为[-1,-2,-3,-4,-5,-6,-7,-8]，则根据selMask的比特位选取后的结果dst_temp为：[9,10,11,12,5,6,7,8]，然后再根据mask进行过滤，dst的最终输出结果为[9,10,11,12,-5,-6,-7,-8]。

本选择功能支持三种模式：

- 模式0：根据selMask在两个tensor中选取元素。selMask中有效数据的个数存在限制，具体取决于源操作数的数据类型。在每一轮迭代中，根据selMask的有效位数据进行选择操作，每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。
- 模式1：根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。多轮迭代时，每轮迭代连续使用selMask的不同部分。
- 模式2：根据selMask在两个tensor中选取元素，selMask无有效数据限制。多轮迭代时，每轮迭代连续使用selMask的不同部分。

Atlas 训练系列产品，仅支持模式0。

Atlas 推理系列产品AI Core，支持模式0、1、2。

Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持模式0、1、2。

Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持模式0、1、2。

Atlas 200I/500 A2 推理产品，支持模式0、1、2。

#### 函数原型

- tensor前n个数据计算Select模式112template<typenameT,typenameU>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,Tsrc1,SELMODEselMode,uint32_tcount)Select模式0和Select模式212template<typenameT,typenameU>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,constLocalTensor<T>&src1,SELMODEselMode,uint32_tcount)
- tensor高维切分计算Select模式1mask逐bit模式12template<typenameT,typenameU,boolisSetMask=true>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,Tsrc1,SELMODEselMode,uint64_tmask[],uint8_trepeatTime,constBinaryRepeatParams&repeatParams)mask连续模式12template<typenameT,typenameU,boolisSetMask=true>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,Tsrc1,SELMODEselMode,uint64_tmask,uint8_trepeatTime,constBinaryRepeatParams&repeatParams)不传入mask参数（需要和SetVectorMask、SetCmpMask(ISASI)配合使用）12template<typenameT,typenameU>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,uint8_trepeatTime,constBinaryRepeatParams&repeatParams)Select模式0和Select模式2mask逐bit模式12template<typenameT,typenameU,boolisSetMask=true>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,constLocalTensor<T>&src1,SELMODEselMode,uint64_tmask[],uint8_trepeatTime,constBinaryRepeatParams&repeatParams)mask连续模式12template<typenameT,typenameU,boolisSetMask=true>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<U>&selMask,constLocalTensor<T>&src0,constLocalTensor<T>&src1,SELMODEselMode,uint64_tmask,uint8_trepeatTime,constBinaryRepeatParams&repeatParams)不传入mask参数（需要和SetVectorMask、SetCmpMask(ISASI)配合使用）12template<typenameT,SELMODEselMode>__aicore__inlinevoidSelect(constLocalTensor<T>&dst,constLocalTensor<T>&src0,constLocalTensor<T>&src1,uint8_trepeatTime,constBinaryRepeatParams&repeatParams)

#### 参数说明

| 参数名称 | 含义 |
| --- | --- |
| T | 源操作数和目的操作数的数据类型。 |
| U | selMask的数据类型。 |
| isSetMask | 保留参数，保持默认值即可。如需使用在接口外部设置mask的功能，可以调用不传入mask参数的接口来实现。 |
| selMode | 同表2 参数说明中的selMode参数说明。 |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。Atlas 训练系列产品，支持的数据类型为：half/floatAtlas 推理系列产品AI Core，支持的数据类型为：half/floatAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/floatAtlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/floatAtlas 200I/500 A2 推理产品，支持的数据类型为：half/float |
| selMask | 输入 | 选取mask。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。支持的数据类型为：uint8_t/uint16_t/uint32_t/uint64_t。每个比特位表示1个元素的选取，当selMask的比特位为1时，从src0中选取元素；比特位为0时，从src1中选取元素。selMode为模式0时，在每一轮迭代中，根据selMask的有效位数据进行选择操作，每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。selMode为模式1/2时，多次迭代对selMask连续消耗。模式0：根据selMask在两个tensor中选取元素，selMask有位数限制，不管迭代多少次，每次迭代都只根据截取后的固定位数的selMask进行选择。selMask的有效位数限制为256/sizeof(T)。模式1：根据selMask在1个tensor和1个scalar标量中选取元素。selMask连续存放，每次迭代使用selMask的位数为256/sizeof(T)。模式2：根据selMask在两个tensor中选取元素。selMask连续存放，每次迭代使用selMask的位数为256/sizeof(T)。 |
| src0 | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。Atlas 训练系列产品，支持的数据类型为：half/floatAtlas 推理系列产品AI Core，支持的数据类型为：half/floatAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/floatAtlas 200I/500 A2 推理产品，支持的数据类型为：half/float |
| src1 | 输入 | 源操作数。当selMode为模式0或模式2时：类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。当selMode为模式1时，类型为T，标量数据类型。Atlas 训练系列产品，支持的数据类型为：half/floatAtlas 推理系列产品AI Core，支持的数据类型为：half/floatAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/floatAtlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/floatAtlas 200I/500 A2 推理产品，支持的数据类型为：half/float |
| selMode | 输入 | 指令模式，SELMODE类型，取值如下：12345enumclassSELMODE:uint8_t{VSEL_CMPMASK_SPR=0,VSEL_TENSOR_SCALAR_MODE,VSEL_TENSOR_TENSOR_MODE,};模式0：取值为VSEL_CMPMASK_SPR。根据selMask在两个tensor中选取元素。selMask中有效数据的个数存在限制，具体取决于源操作数的数据类型。在每一轮迭代中，根据selMask的有效位数据进行选择操作，每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。模式1：取值为VSEL_TENSOR_SCALAR_MODE。根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。多轮迭代时，每轮迭代连续使用selMask的不同部分。模式2：取值为VSEL_TENSOR_TENSOR_MODE。根据selMask在两个tensor中选取元素，selMask无有效数据限制。多轮迭代时，每轮迭代连续使用selMask的不同部分。 | 12345 | enumclassSELMODE:uint8_t{VSEL_CMPMASK_SPR=0,VSEL_TENSOR_SCALAR_MODE,VSEL_TENSOR_TENSOR_MODE,}; |
| 12345 | enumclassSELMODE:uint8_t{VSEL_CMPMASK_SPR=0,VSEL_TENSOR_SCALAR_MODE,VSEL_TENSOR_TENSOR_MODE,}; |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。BinaryRepeatParams类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |
| count | 输入 | 参与计算的元素个数。 |

#### 返回值说明

无

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 操作数地址重叠约束请参考通用地址重叠约束。

- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，对于模式1和模式2，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，对于模式1和模式2，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。
- 针对Atlas 推理系列产品AI Core，对于模式1和模式2，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。

#### 调用示例

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- Select-tensor高维切分计算样例（模式2）1234567uint64_tmask=256/sizeof(float);intrepeat=4;AscendC::BinaryRepeatParamsrepeatParams={1,1,1,8,8,8};// repeat = 4, 64 elements one repeat, 256 elements total// dstBlkStride, src0BlkStride, src1BlkStride = 1, no gap between blocks in one repeat// dstRepStride, src0RepStride, src1RepStride = 8, no gap between repeatsAscendC::Select(dstLocal,maskLocal,src0Local,src1Local,AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,mask,repeat,repeatParams);

- Select-tensor前n个数据计算样例（模式1）1AscendC::Select(dstLocal,maskLocal,src0Local,static_cast<float>(0),AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,dataSize);

- Select-tensor前n个数据计算样例（模式0，每轮repeat迭代中，maskLocal只有前64bit生效）1AscendC::Select(dstLocal,maskLocal,src0Local,src1Local,AscendC::SELMODE::VSEL_CMPMASK_SPR,dataSize);

- Select-tensor高维切分计算样例-mask连续模式（模式0，每轮repeat迭代中，maskLocal只有前64bit生效）1234567uint64_tmask=256/sizeof(float);intrepeat=4;AscendC::BinaryRepeatParamsrepeatParams={1,1,1,8,8,8};// repeat = 4, 64 elements one repeat, 256 elements total// dstBlkStride, src0BlkStride, src1BlkStride = 1, no gap between blocks in one repeat// dstRepStride, src0RepStride, src1RepStride = 8, no gap between repeatsAscendC::Select(dstLocal,maskLocal,src0Local,src1Local,AscendC::SELMODE::VSEL_CMPMASK_SPR,mask,repeat,repeatParams);
- Select-tensor高维切分计算样例-mask逐bit模式（每轮repeat迭代中，maskLocal只有前64bit生效）1234567uint64_tmask[2]={UINT64_MAX,0};intrepeat=4;AscendC::BinaryRepeatParamsrepeatParams={1,1,1,8,8,8};// repeat = 4, 64 elements one repeat, 256 elements total// srcBlkStride, = 1, no gap between blocks in one repeat// dstRepStride, srcRepStride = 8, no gap between repeatsAscendC::Select(dstLocal,maskLocal,src0Local,src1Local,AscendC::SELMODE::VSEL_CMPMASK_SPR,mask,repeat,repeatParams);

结果示例如下：

模式2示例：

模式1示例：

#### 样例模板

| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475 | #include"kernel_operator.h"classKernelSelect{public:__aicore__inlineKernelSelect(){}__aicore__inlinevoidInit(__gm__uint8_t*src0Gm,__gm__uint8_t*src1Gm,__gm__uint8_t*selGm,__gm__uint8_t*dstGm){src0Global.SetGlobalBuffer((__gm__float*)src0Gm);src1Global.SetGlobalBuffer((__gm__float*)src1Gm);selMaskGlobal.SetGlobalBuffer((__gm__uint8_t*)selGm);dstGlobal.SetGlobalBuffer((__gm__float*)dstGm);pipe.InitBuffer(inQueueSrc0,1,dataSize*sizeof(float));pipe.InitBuffer(inQueueSrc1,1,dataSize*sizeof(float));pipe.InitBuffer(inQueueSelMask,1,selDataSize*sizeof(uint8_t));pipe.InitBuffer(outQueueDst,1,dataSize*sizeof(float));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<float>src0Local=inQueueSrc0.AllocTensor<float>();AscendC::LocalTensor<float>src1Local=inQueueSrc1.AllocTensor<float>();AscendC::LocalTensor<uint8_t>selMaskLocal=inQueueSelMask.AllocTensor<uint8_t>();AscendC::DataCopy(src0Local,src0Global,dataSize);AscendC::DataCopy(src1Local,src1Global,dataSize);AscendC::DataCopy(selMaskLocal,selMaskGlobal,selDataSize);inQueueSrc0.EnQue(src0Local);inQueueSrc1.EnQue(src1Local);inQueueSelMask.EnQue(selMaskLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<float>src0Local=inQueueSrc0.DeQue<float>();AscendC::LocalTensor<float>src1Local=inQueueSrc1.DeQue<float>();AscendC::LocalTensor<uint8_t>maskLocal=inQueueSelMask.DeQue<uint8_t>();AscendC::LocalTensor<float>dstLocal=outQueueDst.AllocTensor<float>();AscendC::Select(dstLocal,maskLocal,src0Local,src1Local,AscendC::SELMODE::VSEL_CMPMASK_SPR,dataSize);outQueueDst.EnQue<float>(dstLocal);inQueueSrc0.FreeTensor(src0Local);inQueueSrc1.FreeTensor(src1Local);inQueueSelMask.FreeTensor(maskLocal);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<float>dstLocal=outQueueDst.DeQue<float>();AscendC::DataCopy(dstGlobal,dstLocal,dataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc0,inQueueSrc1,inQueueSelMask;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<float>src0Global,src1Global,dstGlobal;AscendC::GlobalTensor<uint8_t>selMaskGlobal;uint32_tdataSize=256;uint32_toneSelectDataSize=256/sizeof(float);uint32_tselDataSize=dataSize/oneSelectDataSize*32;// （模式1和模式2时，uint32_t selDataSize = dataSize / 8;）};extern"C"__global____aicore__voidmain_sel_demo(__gm__uint8_t*src0Gm,__gm__uint8_t*src1Gm,__gm__uint8_t*selGm,__gm__uint8_t*dstGm){KernelSelectop;op.Init(src0Gm,src1Gm,selGm,dstGm);op.Process();} |
| --- | --- |
