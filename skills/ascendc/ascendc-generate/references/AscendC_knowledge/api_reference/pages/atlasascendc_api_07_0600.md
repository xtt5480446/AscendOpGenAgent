# Xor-Xor接口-数学计算-高阶API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0600
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0600.html
---

# Xor

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

按元素执行Xor运算，Xor（异或）的概念和运算规则如下：

- 概念：参加运算的两个数据，按二进制位进行“异或”运算。
- 运算规则：0^0=0；0^1=1；1^0=1；1^1=0；即：参加运算的两个对象，如果两个相应位为“异”（值不同），则该位结果为1，否则为 0【同0异1】。

计算公式如下：

![](../images/atlasascendc_api_07_0601_img_001.png)

![](../images/atlasascendc_api_07_0601_img_002.png)

| 1 | 例如：3^5=6，即00000011^00000101=00000110 |
| --- | --- |

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间源操作数Tensor全部/部分参与计算12template<typenameT,boolisReuseSource=false>__aicore__inlinevoidXor(constLocalTensor<T>&dstTensor,constLocalTensor<T>&src0Tensor,constLocalTensor<T>&src1Tensor,constLocalTensor<uint8_t>&sharedTmpBuffer,constuint32_tcalCount)源操作数Tensor全部参与计算12template<typenameT,boolisReuseSource=false>__aicore__inlinevoidXor(constLocalTensor<T>&dstTensor,constLocalTensor<T>&src0Tensor,constLocalTensor<T>&src1Tensor,constLocalTensor<uint8_t>&sharedTmpBuffer)

- 接口框架申请临时空间源操作数Tensor全部/部分参与计算12template<typenameT,boolisReuseSource=false>__aicore__inlinevoidXor(constLocalTensor<T>&dstTensor,constLocalTensor<T>&src0Tensor,constLocalTensor<T>&src1Tensor,constuint32_tcalCount)源操作数Tensor全部参与计算12template<typenameT,boolisReuseSource=false>__aicore__inlinevoidXor(constLocalTensor<T>&dstTensor,constLocalTensor<T>&src0Tensor,constLocalTensor<T>&src1Tensor)

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者通过sharedTmpBuffer入参传入和接口框架申请两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetXorMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t、uint16_t。Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t、uint16_t。Atlas 200I/500 A2 推理产品，支持的数据类型为：int16_t、uint16_t。Atlas 推理系列产品AI Core，支持的数据类型为：int16_t、uint16_t。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0Tensor | 输入 | 源操作数0。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。 |
| src1Tensor | 输入 | 源操作数1。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。用于Xor内部复杂计算时存储中间变量，由开发者提供。临时空间大小BufferSize的获取方式请参考GetXorMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 返回值说明

无

#### 约束说明

- 不支持源操作数与目的操作数地址重叠。
- 当前仅支持ND格式的输入，不支持其他格式。
- calCount需要保证小于或等于src0Tensor和src1Tensor和dstTensor存储的元素范围。
- 对于不带calCount参数的接口，需要保证src0Tensor和src1Tensor的shape大小相等。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
- 操作数地址对齐要求请参见通用地址对齐约束。

#### 调用示例

调用样例kernel侧xor_custom.cpp

| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100101102103104105106107 | #include"kernel_operator.h"constexprint32_tBUFFER_NUM=1;classKernelXor{public:__aicore__inlineKernelXor(){}__aicore__inlinevoidInit(GM_ADDRx,GM_ADDRy,GM_ADDRz,uint32_ttotalLength,uint32_ttotalLength2,uint32_ttilenum,uint32_ttmpSize,uint32_tmcount){this->totalLength=totalLength;this->blockLength=totalLength/AscendC::GetBlockNum();this->blockLength2=totalLength2/AscendC::GetBlockNum();this->tilenum=tilenum;this->tmpSize=tmpSize;this->mcount=mcount;this->tileLength=this->blockLength/tilenum/BUFFER_NUM;this->tileLength2=this->blockLength2/tilenum/BUFFER_NUM;xGm.SetGlobalBuffer((__gm__int16_t*)x+this->blockLength*AscendC::GetBlockIdx(),this->blockLength);yGm.SetGlobalBuffer((__gm__int16_t*)y+this->blockLength2*AscendC::GetBlockIdx(),this->blockLength2);zGm.SetGlobalBuffer((__gm__int16_t*)z+this->blockLength*AscendC::GetBlockIdx(),this->blockLength);if(this->tmpSize!=0){pipe.InitBuffer(tmpQueue,BUFFER_NUM,this->tmpSize);}pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength*sizeof(int16_t));pipe.InitBuffer(inQueueY,BUFFER_NUM,this->tileLength2*sizeof(int16_t));pipe.InitBuffer(outQueueZ,BUFFER_NUM,this->tileLength*sizeof(int16_t));}__aicore__inlinevoidProcess(){int32_tloopCount=this->tilenum*BUFFER_NUM;for(int32_ti=0;i<loopCount;i++){CopyIn(i);Compute(i);CopyOut(i);}}private:__aicore__inlinevoidCopyIn(int32_tprogress){AscendC::LocalTensor<int16_t>xLocal=inQueueX.AllocTensor<int16_t>();AscendC::DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);inQueueX.EnQue(xLocal);AscendC::LocalTensor<int16_t>yLocal=inQueueY.AllocTensor<int16_t>();AscendC::DataCopy(yLocal,yGm[progress*this->tileLength2],this->tileLength2);inQueueY.EnQue(yLocal);}__aicore__inlinevoidCompute(int32_tprogress){AscendC::LocalTensor<int16_t>xLocal=inQueueX.DeQue<int16_t>();AscendC::LocalTensor<int16_t>yLocal=inQueueY.DeQue<int16_t>();AscendC::LocalTensor<int16_t>zLocal=outQueueZ.AllocTensor<int16_t>();if(this->tmpSize!=0){AscendC::LocalTensor<uint8_t>tmpLocal=tmpQueue.AllocTensor<uint8_t>();if(this->mcount!=this->totalLength){AscendC::Xor(zLocal,xLocal,yLocal,tmpLocal,this->mcount);}else{AscendC::Xor(zLocal,xLocal,yLocal,tmpLocal);}tmpQueue.FreeTensor(tmpLocal);}else{if(this->mcount!=this->totalLength){AscendC::Xor(zLocal,xLocal,yLocal,this->mcount);}else{AscendC::Xor(zLocal,xLocal,yLocal);}}outQueueZ.EnQue<int16_t>(zLocal);inQueueX.FreeTensor(xLocal);inQueueY.FreeTensor(yLocal);}__aicore__inlinevoidCopyOut(int32_tprogress){AscendC::LocalTensor<int16_t>zLocal=outQueueZ.DeQue<int16_t>();AscendC::DataCopy(zGm[progress*this->tileLength],zLocal,this->tileLength);outQueueZ.FreeTensor(zLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,BUFFER_NUM>inQueueX;AscendC::TQue<AscendC::TPosition::VECIN,BUFFER_NUM>inQueueY;AscendC::TQue<AscendC::TPosition::VECIN,BUFFER_NUM>tmpQueue;AscendC::TQue<AscendC::TPosition::VECOUT,BUFFER_NUM>outQueueZ;AscendC::GlobalTensor<int16_t>xGm;AscendC::GlobalTensor<int16_t>yGm;AscendC::GlobalTensor<int16_t>zGm;uint32_tblockLength;uint32_tblockLength2;uint32_ttilenum;uint32_ttileLength;uint32_ttileLength2;uint32_ttmpSize;uint32_tmcount;uint32_ttotalLength;};extern"C"__global____aicore__voidxor_custom(GM_ADDRx,GM_ADDRy,GM_ADDRz,GM_ADDRworkspace,GM_ADDRtiling){GET_TILING_DATA(tilingData,tiling);KernelXorop;op.Init(x,y,z,tilingData.totalLength,tilingData.totalLength2,tilingData.tilenum,tilingData.tmpSize,tilingData.mcount);if(TILING_KEY_IS(1)){op.Process();}} |
| --- | --- |

host侧xor_custom_tiling.h

| 123456789101112 | #include"register/op_def_registry.h"#include"register/tilingdata_base.h"namespaceoptiling{BEGIN_TILING_DATA_DEF(XorCustomTilingData)TILING_DATA_FIELD_DEF(uint32_t,totalLength);TILING_DATA_FIELD_DEF(uint32_t,totalLength2);TILING_DATA_FIELD_DEF(uint32_t,tmpSize);TILING_DATA_FIELD_DEF(uint32_t,tilenum);TILING_DATA_FIELD_DEF(uint32_t,mcount);END_TILING_DATA_DEF;REGISTER_TILING_DATA_CLASS(XorCustom,XorCustomTilingData)} |
| --- | --- |

host侧xor_custom.cpp

| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899 | #include"xor_custom_tiling.h"#include"register/op_def_registry.h"#include"tiling/tiling_api.h"namespaceoptiling{staticge::graphStatusTilingFunc(gert::TilingContext*context){XorCustomTilingDatatiling;constgert::RuntimeAttrs*xorAttrs=context->GetAttrs();constuint32_ttilenum=*(xorAttrs->GetAttrPointer<uint32_t>(0));constuint32_tblockdim=*(xorAttrs->GetAttrPointer<uint32_t>(1));constuint32_tsizeflag=*(xorAttrs->GetAttrPointer<uint32_t>(2));constuint32_tcountflag=*(xorAttrs->GetAttrPointer<uint32_t>(3));uint32_ttotalLength=context->GetInputTensor(0)->GetShapeSize();uint32_ttotalLength2=context->GetInputTensor(1)->GetShapeSize();context->SetBlockDim(blockdim);tiling.set_totalLength(totalLength);tiling.set_totalLength2(totalLength2);tiling.set_tilenum(tilenum);if(countflag==0){tiling.set_mcount(totalLength2);}elseif(countflag==1){tiling.set_mcount(totalLength);}std::vector<int64_t>shapeVec={totalLength};ge::ShapesrcShape(shapeVec);uint32_ttypeSize=sizeof(int16_t);uint32_tmaxValue=0;uint32_tminValue=0;boolisReuseSource=false;AscendC::GetXorMaxMinTmpSize(srcShape,typeSize,isReuseSource,maxValue,minValue);// sizeflag 0：代表取最小的tempBuffer 1：取最大的tempBufferif(sizeflag==0){tiling.set_tmpSize(minValue);}elseif(sizeflag==1){tiling.set_tmpSize(maxValue);}elseif(sizeflag==2){tiling.set_tmpSize(0);}tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());context->SetTilingKey(1);size_t*currentWorkspace=context->GetWorkspaceSizes(1);currentWorkspace[0]=0;returnge::GRAPH_SUCCESS;}}namespacege{staticge::graphStatusInferShape(gert::InferShapeContext*context){constgert::Shape*xShape=context->GetInputShape(0);gert::Shape*yShape=context->GetOutputShape(0);*yShape=*xShape;returnGRAPH_SUCCESS;}}namespaceops{classXorCustom:publicOpDef{public:explicitXorCustom(constchar*name):OpDef(name){this->Input("x").ParamType(REQUIRED).DataType({ge::DT_INT16}).Format({ge::FORMAT_ND});this->Input("y").ParamType(REQUIRED).DataType({ge::DT_INT16}).Format({ge::FORMAT_ND});this->Output("z").ParamType(REQUIRED).DataType({ge::DT_INT16}).Format({ge::FORMAT_ND});this->SetInferShape(ge::InferShape);this->Attr("tilenum").AttrType(REQUIRED).Int(0);this->Attr("blockdim").AttrType(REQUIRED).Int(0);this->Attr("sizeflag").AttrType(REQUIRED).Int(0);this->Attr("countflag").AttrType(REQUIRED).Int(0);this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascendxxx");// ascendxxx请修改为对应的昇腾AI处理器型号。}};OP_ADD(XorCustom);}// namespace ops |
| --- | --- |

| 1234 | 输入输出的数据类型为int16_t，一维向量包含32个数。例如向量中第一个数据进行异或：(-5753)xor18745=-24386输入数据(src0Local):[-57532850120334-5845...-2081734032126122241]输入数据(src1Local):[18745-244482087310759...21940-26342925131019]输出数据(dstLocal):[-24386-123317911-15572...-1253-275673051012234] |
| --- | --- |
