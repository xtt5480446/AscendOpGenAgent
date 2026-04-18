# CompareScalar-比较与选择-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0068
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0068.html
---

# CompareScalar

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

逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

支持多种比较模式：

- LT：小于（less than）
- GT：大于（greater than）

- GE：大于或等于（greater than or equal to）
- EQ：等于（equal to）
- NE：不等于（not equal to）
- LE：小于或等于（less than or equal to）

#### 函数原型

- tensor前n个数据计算12template<typenameT,typenameU>__aicore__inlinevoidCompareScalar(constLocalTensor<U>&dst,constLocalTensor<T>&src0,constTsrc1Scalar,CMPMODEcmpMode,uint32_tcount)
- tensor高维切分计算mask逐bit模式12template<typenameT,typenameU,boolisSetMask=true>__aicore__inlinevoidCompareScalar(constLocalTensor<U>&dst,constLocalTensor<T>&src0,constTsrc1Scalar,CMPMODEcmpMode,constuint64_tmask[],uint8_trepeatTime,constUnaryRepeatParams&repeatParams)mask连续模式12template<typenameT,typenameU,boolisSetMask=true>__aicore__inlinevoidCompareScalar(constLocalTensor<U>&dst,constLocalTensor<T>&src0,constTsrc1Scalar,CMPMODEcmpMode,constuint64_tmask,uint8_trepeatTime,constUnaryRepeatParams&repeatParams)

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数数据类型。 |
| U | 目的操作数数据类型。 |
| isSetMask | 是否在接口内部设置mask。true，表示在接口内部设置mask。false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。dst用于存储比较结果，将dst中uint8_t类型的数据按照bit位展开，由左至右依次表征对应位置的src0和src1的比较结果，如果比较后的结果为真，则对应比特位为1，否则为0。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：uint8_tAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：uint8_tAtlas 200I/500 A2 推理产品，支持的数据类型为：uint8_tAtlas 推理系列产品AI Core，支持的数据类型为：uint8_t |
| src0 | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float（所有CMPMODE都支持）， int32_t（只支持CMPMODE::EQ）Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float（所有CMPMODE都支持）， int32_t（只支持CMPMODE::EQ）Atlas 200I/500 A2 推理产品，支持的数据类型为：int16_t/uint16_t/half/float/int32_t/uint32_tAtlas 推理系列产品AI Core，支持的数据类型为：half/float |
| src1Scalar | 输入 | 源操作数，Scalar标量。数据类型和src0保持一致。 |
| cmpMode | 输入 | CMPMODE类型，表示比较模式，包括EQ，NE，GE，LE，GT，LT。LT： src0小于（less than）src1GT： src0大于（greater than）src1GE：src0大于或等于（greater than or equal to）src1EQ：src0等于（equal to）src1NE：src0不等于（not equal to）src1LE：src0小于或等于（less than or equal to）src1 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。Atlas A3 训练系列产品/Atlas A3 推理系列产品，保留参数，设置无效。Atlas A2 训练系列产品/Atlas A2 推理系列产品，保留参数，设置无效。Atlas 200I/500 A2 推理产品，设置有效。Atlas 推理系列产品AI Core，保留参数，设置无效。连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]。逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。参数类型为长度为2或者4的uint64_t类型数组。例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 264-1]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。UnaryRepeatParams类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。相邻迭代间的地址步长参数说明请参考repeatStride；同一迭代内DataBlock的地址步长参数说明请参考dataBlockStride。 |
| count | 输入 | 参与计算的元素个数。设置count时，需要保证count个元素所占空间256字节对齐。 |

#### 返回值说明

无

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。

- dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
- 使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。

#### 调用示例

本样例中，源操作数src0Local存储了256个float类型的数据。样例实现的功能为，对src0Local中的元素和src1Local.GetValue(0)中的数据进行比较，如果src0Local中的元素小于src1Local.GetValue(0)中的元素，dstLocal结果中对应的比特位置1；反之，则置0。dst结果使用uint8_t类型数据存储。

- tensor前n个数据计算接口样例1AscendC::CompareScalar(dstLocal,src0Local,src1Scalar,AscendC::CMPMODE::LT,srcDataSize);

- tensor高维切分计算-mask连续模式1234567uint64_tmask=256/sizeof(float);// 256为每个迭代处理的字节数intrepeat=4;AscendC::UnaryRepeatParamsrepeatParams={1,1,8,8};// repeat = 4, 64 elements one repeat, 256 elements total// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat// dstRepStride, srcRepStride = 8, no gap between repeatsAscendC::CompareScalar(dstLocal,src0Local,src1Scalar,AscendC::CMPMODE::LT,mask,repeat,repeatParams);
- tensor高维切分计算-mask逐bit模式1234567uint64_tmask[2]={UINT64_MAX,0};intrepeat=4;AscendC::UnaryRepeatParamsrepeatParams={1,1,8,8};// repeat = 4, 64 elements one repeat, 256 elements total// srcBlkStride, = 1, no gap between blocks in one repeat// dstRepStride, srcRepStride = 8, no gap between repeatsAscendC::CompareScalar(dstLocal,src0Local,src1Scalar,AscendC::CMPMODE::LT,mask,repeat,repeatParams);

#### 样例模板

| 12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455565758596061626364656667686970717273 | #include"kernel_operator.h"template<typenameT>classKernelCmp{public:__aicore__inlineKernelCmp(){}__aicore__inlinevoidInit(__gm__uint8_t*src0Gm,__gm__uint8_t*src1Gm,__gm__uint8_t*dstGm,uint32_tdataSize,AscendC::CMPMODEmode){srcDataSize=dataSize;dstDataSize=srcDataSize/8;cmpMode=mode;src0Global.SetGlobalBuffer((__gm__T*)src0Gm);src1Global.SetGlobalBuffer((__gm__T*)src1Gm);dstGlobal.SetGlobalBuffer((__gm__uint8_t*)dstGm);pipe.InitBuffer(inQueueSrc0,1,srcDataSize*sizeof(T));pipe.InitBuffer(inQueueSrc1,1,16*sizeof(T));pipe.InitBuffer(outQueueDst,1,dstDataSize*sizeof(uint8_t));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<T>src0Local=inQueueSrc0.AllocTensor<T>();AscendC::LocalTensor<T>src1Local=inQueueSrc1.AllocTensor<T>();AscendC::DataCopy(src0Local,src0Global,srcDataSize);AscendC::DataCopy(src1Local,src1Global,16);inQueueSrc0.EnQue(src0Local);inQueueSrc1.EnQue(src1Local);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<T>src0Local=inQueueSrc0.DeQue<T>();AscendC::LocalTensor<T>src1Local=inQueueSrc1.DeQue<T>();AscendC::LocalTensor<uint8_t>dstLocal=outQueueDst.AllocTensor<uint8_t>();AscendC::PipeBarrier<PIPE_ALL>();Tsrc1Scalar=src1Local.GetValue(0);AscendC::PipeBarrier<PIPE_ALL>();AscendC::CompareScalar(dstLocal,src0Local,static_cast<T>(src1Scalar),cmpMode,srcDataSize);outQueueDst.EnQue<uint8_t>(dstLocal);inQueueSrc0.FreeTensor(src0Local);inQueueSrc1.FreeTensor(src1Local);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<uint8_t>dstLocal=outQueueDst.DeQue<uint8_t>();AscendC::DataCopy(dstGlobal,dstLocal,dstDataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc0,inQueueSrc1;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<T>src0Global,src1Global;AscendC::GlobalTensor<uint8_t>dstGlobal;uint32_tsrcDataSize=0;uint32_tdstDataSize=0;AscendC::CMPMODEcmpMode;};template<typenameT>__aicore__voidmain_cpu_cmp_sel_demo(__gm__uint8_t*src0Gm,__gm__uint8_t*src1Gm,__gm__uint8_t*dstGm,uint32_tdataSize,AscendC::CMPMODEmode){KernelCmp<T>op;op.Init(src0Gm,src1Gm,dstGm,dataSize,mode);op.Process();}extern"C"__global____aicore__voidkernel_vec_compare_scalar_256_LT_float(GM_ADDRsrc0_gm,GM_ADDRsrc1_gm,GM_ADDRdst_gm){main_cpu_cmp_sel_demo<float>(src0_gm,src1_gm,dst_gm,256,AscendC::CMPMODE::LT);} |
| --- | --- |
