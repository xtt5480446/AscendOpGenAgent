# Compare（结果存入寄存器）-比较与选择-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0067
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0067.html
---

# Compare（结果存入寄存器）

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

逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。Compare接口需要mask参数时，可以使用此接口。计算结果存入寄存器中。

支持多种比较模式：

- LT：小于（less than）
- GT：大于（greater than）

- GE：大于或等于（greater than or equal to）
- EQ：等于（equal to）
- NE：不等于（not equal to）
- LE：小于或等于（less than or equal to）

#### 函数原型

- mask逐bit模式12template<typenameT,boolisSetMask=true>__aicore__inlinevoidCompare(constLocalTensor<T>&src0,constLocalTensor<T>&src1,CMPMODEcmpMode,constuint64_tmask[],constBinaryRepeatParams&repeatParams)
- mask连续模式12template<typenameT,boolisSetMask=true>__aicore__inlinevoidCompare(constLocalTensor<T>&src0,constLocalTensor<T>&src1,CMPMODEcmpMode,constuint64_tmask,constBinaryRepeatParams&repeatParams)

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数数据类型。Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/floatAtlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/floatAtlas 推理系列产品AI Core，支持的数据类型为：half/float |
| isSetMask | 是否在接口内部设置mask。true，表示在接口内部设置mask。false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| src0、src1 | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。 |
| cmpMode | 输入 | CMPMODE类型，表示比较模式，包括EQ，NE，GE，LE，GT，LT。LT： src0小于（less than）src1GT： src0大于（greater than）src1GE：src0大于或等于（greater than or equal to）src1EQ：src0等于（equal to）src1NE：src0不等于（not equal to）src1LE：src0小于或等于（less than or equal to）src1 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。BinaryRepeatParams类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |

#### 返回值说明

无

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。

- 本接口没有repeat输入，repeat默认为1，即一条指令计算256B的数据。
- 本接口将结果写入128bit的cmpMask寄存器中，可以用GetCmpMask接口获取寄存器保存的数据。

#### 调用示例

本样例中，源操作数src0Local和src1Local各存储了64个float类型的数据。样例实现的功能为，逐元素对src0Local和src1Local中的数据进行比较，如果src0Local中的元素小于src1Local中的元素，dstLocal结果中对应的比特位置1；反之，则置0。dstLocal结果使用uint8_t类型数据存储。

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- mask连续模式12345uint64_tmask=256/sizeof(float);// 256为每个迭代处理的字节数AscendC::BinaryRepeatParamsrepeatParams={1,1,1,8,8,8};// dstBlkStride, src0BlkStride, src1BlkStride = 1, no gap between blocks in one repeat// dstRepStride, src0RepStride, src1RepStride = 8, no gap between repeatsAscendC::Compare(src0Local,src1Local,AscendC::CMPMODE::LT,mask,repeatParams);
- mask逐bit模式12345uint64_tmask[2]={UINT64_MAX,0};AscendC::BinaryRepeatParamsrepeatParams={1,1,1,8,8,8};// srcBlkStride, = 1, no gap between blocks in one repeat// dstRepStride, srcRepStride = 8, no gap between repeatsAscendC::Compare(src0Local,src1Local,AscendC::CMPMODE::LT,mask,repeatParams);

#### 样例模板

| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475 | #include"kernel_operator.h"template<typenameT>classKernelCmpCmpmask{public:__aicore__inlineKernelCmpCmpmask(){}__aicore__inlinevoidInit(__gm__uint8_t*src0Gm,__gm__uint8_t*src1Gm,__gm__uint8_t*dstGm,uint32_tdataSize,AscendC::CMPMODEmode){srcDataSize=dataSize;dstDataSize=32;cmpMode=mode;src0Global.SetGlobalBuffer((__gm__T*)src0Gm);src1Global.SetGlobalBuffer((__gm__T*)src1Gm);dstGlobal.SetGlobalBuffer((__gm__uint8_t*)dstGm);pipe.InitBuffer(inQueueSrc0,1,srcDataSize*sizeof(T));pipe.InitBuffer(inQueueSrc1,1,srcDataSize*sizeof(T));pipe.InitBuffer(outQueueDst,1,dstDataSize*sizeof(uint8_t));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<T>src0Local=inQueueSrc0.AllocTensor<T>();AscendC::LocalTensor<T>src1Local=inQueueSrc1.AllocTensor<T>();AscendC::DataCopy(src0Local,src0Global,srcDataSize);AscendC::DataCopy(src1Local,src1Global,srcDataSize);inQueueSrc0.EnQue(src0Local);inQueueSrc1.EnQue(src1Local);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<T>src0Local=inQueueSrc0.DeQue<T>();AscendC::LocalTensor<T>src1Local=inQueueSrc1.DeQue<T>();AscendC::LocalTensor<uint8_t>dstLocal=outQueueDst.AllocTensor<uint8_t>();AscendC::Duplicate(dstLocal.ReinterpretCast<float>(),static_cast<float>(0),8);AscendC::BinaryRepeatParamsrepeatParams;uint32_tmask=256/sizeof(T);AscendC::Compare(src0Local,src1Local,cmpMode,mask,repeatParams);AscendC::PipeBarrier<PIPE_V>();AscendC::GetCmpMask(dstLocal);outQueueDst.EnQue<uint8_t>(dstLocal);inQueueSrc0.FreeTensor(src0Local);inQueueSrc1.FreeTensor(src1Local);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<uint8_t>dstLocal=outQueueDst.DeQue<uint8_t>();AscendC::DataCopy(dstGlobal,dstLocal,dstDataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc0,inQueueSrc1;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<T>src0Global,src1Global;AscendC::GlobalTensor<uint8_t>dstGlobal;uint32_tsrcDataSize=0;uint32_tdstDataSize=0;AscendC::CMPMODEcmpMode;};template<typenameT>__aicore__voidmain_cpu_cmp_cmpmask_demo(__gm__uint8_t*src0Gm,__gm__uint8_t*src1Gm,__gm__uint8_t*dstGm,uint32_tdataSize,AscendC::CMPMODEmode){KernelCmpCmpmask<T>op;op.Init(src0Gm,src1Gm,dstGm,dataSize,mode);op.Process();}extern"C"__global____aicore__voidkernel_vec_compare_cmpmask_64_LT_float(GM_ADDRsrc0_gm,GM_ADDRsrc1_gm,GM_ADDRdst_gm){main_cpu_cmp_cmpmask_demo<float>(src0_gm,src1_gm,dst_gm,64,AscendC::CMPMODE::LT);} |
| --- | --- |
