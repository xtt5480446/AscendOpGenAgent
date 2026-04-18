# Load2D-LoadData-数据搬运-矩阵计算（ISASI）-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0238
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0238.html
---

# Load2D

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

Load2D支持如下数据通路的搬运：

GM->A1; GM->B1; GM->A2; GM->B2;

A1->A2; B1->B2。

#### 函数原型

- Load2D接口1234template<typenameT>__aicore__inlinevoidLoadData(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLoadData2DParams&loadDataParams)template<typenameT>__aicore__inlinevoidLoadData(constLocalTensor<T>&dst,constGlobalTensor<T>&src,constLoadData2DParams&loadDataParams)

#### 参数说明

| 参数名称 | 含义 |
| --- | --- |
| T | 源操作数和目的操作数的数据类型。Load2D接口Atlas 训练系列产品，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/halfAtlas 推理系列产品AI Core，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/halfAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/floatAtlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/floatAtlas 200I/500 A2 推理产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，类型为LocalTensor。数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下：A2：ZZ格式；对应的分形大小为16 * (32B / sizeof(T))。B2：ZN格式；对应的分形大小为 (32B / sizeof(T))  * 16。A1/B1：无格式要求，一般情况下为NZ格式。NZ格式下，对应的分形大小为16 * (32B / sizeof(T))。 |
| src | 输入 | 源操作数，类型为LocalTensor或GlobalTensor。数据类型需要与dst保持一致。 |
| loadDataParams | 输入 | LoadData参数结构体，类型为：LoadData2DParams，具体参考表3。上述结构体参数定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 |

| 参数名称 | 含义 |
| --- | --- |
| startIndex | 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。 |
| repeatTimes | 迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。 |
| srcStride | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。 |
| sid | 预留参数，配置为0即可。 |
| dstGap | 相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。注：Atlas 训练系列产品此参数不使能。 |
| ifTranspose | 是否启用转置功能，对每个分形矩阵进行转置，默认为false:true：启用false：不启用注意：只有A1->A2和B1->B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型。 |
| addrMode | 预留参数，配置为0即可。 |

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。

#### 返回值说明

无

#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core。

| 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100101102103104105106107108109110111112113114115116117118119120121122123124125126127128129130131132133134135136137138 | #include"kernel_operator.h"classKernelLoadData{public:__aicore__inlineKernelLoadData(){coutBlocks=(Cout+16-1)/16;ho=(H+padTop+padBottom-dilationH*(Kh-1)-1)/strideH+1;wo=(W+padLeft+padRight-dilationW*(Kw-1)-1)/strideW+1;howo=ho*wo;howoRound=((howo+16-1)/16)*16;featureMapA1Size=C1*H*W*C0;// shape: [C1, H, W, C0]weightA1Size=C1*Kh*Kw*Cout*C0;// shape: [C1, Kh, Kw, Cout, C0]featureMapA2Size=howoRound*(C1*Kh*Kw*C0);weightB2Size=(C1*Kh*Kw*C0)*coutBlocks*16;m=howo;k=C1*Kh*Kw*C0;n=Cout;dstSize=coutBlocks*howo*16;// shape: [coutBlocks, howo, 16]dstCO1Size=coutBlocks*howoRound*16;fmRepeat=featureMapA2Size/(16*C0);weRepeat=weightB2Size/(16*C0);}__aicore__inlinevoidInit(__gm__uint8_t*fmGm,__gm__uint8_t*weGm,__gm__uint8_t*dstGm){fmGlobal.SetGlobalBuffer((__gm__half*)fmGm);weGlobal.SetGlobalBuffer((__gm__half*)weGm);dstGlobal.SetGlobalBuffer((__gm__half*)dstGm);pipe.InitBuffer(inQueueFmA1,1,featureMapA1Size*sizeof(half));pipe.InitBuffer(inQueueFmA2,1,featureMapA2Size*sizeof(half));pipe.InitBuffer(inQueueWeB1,1,weightA1Size*sizeof(half));pipe.InitBuffer(inQueueWeB2,1,weightB2Size*sizeof(half));pipe.InitBuffer(outQueue,1,dstCO1Size*sizeof(float));pipe.InitBuffer(outQueueUB,1,dstSize*sizeof(half));}__aicore__inlinevoidProcess(){CopyIn();Split();Compute();CopyUB();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<half>featureMapA1=inQueueFmA1.AllocTensor<half>();AscendC::LocalTensor<half>weightB1=inQueueWeB1.AllocTensor<half>();AscendC::DataCopy(featureMapA1,fmGlobal,{1,static_cast<uint16_t>(featureMapA1Size*sizeof(half)/32),0,0});AscendC::DataCopy(weightB1,weGlobal,{1,static_cast<uint16_t>(weightA1Size*sizeof(half)/32),0,0});inQueueFmA1.EnQue(featureMapA1);inQueueWeB1.EnQue(weightB1);}__aicore__inlinevoidSplit(){AscendC::LocalTensor<half>featureMapA1=inQueueFmA1.DeQue<half>();AscendC::LocalTensor<half>weightB1=inQueueWeB1.DeQue<half>();AscendC::LocalTensor<half>featureMapA2=inQueueFmA2.AllocTensor<half>();AscendC::LocalTensor<half>weightB2=inQueueWeB2.AllocTensor<half>();uint8_tpadList[4]={padLeft,padRight,padTop,padBottom};AscendC::LoadData(featureMapA2,featureMapA1,{padList,H,W,0,0,0,-1,-1,strideW,strideH,Kw,Kh,dilationW,dilationH,1,0,fmRepeat,0,(half)(0)});AscendC::LoadData(weightB2,weightB1,{0,weRepeat,1,0,0,false,0});inQueueFmA2.EnQue<half>(featureMapA2);inQueueWeB2.EnQue<half>(weightB2);inQueueFmA1.FreeTensor(featureMapA1);inQueueWeB1.FreeTensor(weightB1);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<half>featureMapA2=inQueueFmA2.DeQue<half>();AscendC::LocalTensor<half>weightB2=inQueueWeB2.DeQue<half>();AscendC::LocalTensor<float>dstCO1=outQueueCO1.AllocTensor<float>();AscendC::Mmad(dstCO1,featureMapA2,weightB2,{m,n,k,0,false,true});outQueueCO1.EnQue<float>(dstCO1);inQueueFmA2.FreeTensor(featureMapA2);inQueueWeB2.FreeTensor(weightB2);}__aicore__inlinevoidCopyUB(){AscendC::LocalTensor<float>dstCO1=outQueueCO1.DeQue<float>();AscendC::LocalTensor<half>dstUB=outQueueUB.AllocTensor<half>();AscendC::DataCopyParamsdataCopyParams;dataCopyParams.blockCount=1;dataCopyParams.blockLen=m*n*sizeof(float)/1024;AscendC::DataCopyEnhancedParamsenhancedParams;enhancedParams.blockMode=AscendC::BlockMode::BLOCK_MODE_MATRIX;AscendC::DataCopy(dstUB,dstCO1,dataCopyParams,enhancedParams);outQueueUB.EnQue<half>(dstUB);outQueueCO1.FreeTensor(dstCO1);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<half>dstUB=outQueueUB.DeQue<half>();AscendC::DataCopy(dstGlobal,dstUB,m*n);outQueueUB.FreeTensor(dstUB);}private:AscendC::TPipepipe;// feature map queueAscendC::TQue<AscendC::TPosition::A1,1>inQueueFmA1;AscendC::TQue<AscendC::TPosition::A2,1>inQueueFmA2;// weight queueAscendC::TQue<AscendC::TPosition::B1,1>inQueueWeB1;AscendC::TQue<AscendC::TPosition::B2,1>inQueueWeB2;// dst queueAscendC::TQue<AscendC::TPosition::CO1,1>outQueueCO1;AscendC::TQue<AscendC::TPosition::CO2,1>outQueueUB;AscendC::GlobalTensor<half>fmGlobal,weGlobal,dstGlobal;uint16_tC1=2;uint16_tH=4,W=4;uint8_tKh=2,Kw=2;uint16_tCout=16;uint16_tC0=16;uint8_tdilationH=2,dilationW=2;uint8_tpadTop=1,padBottom=1,padLeft=1,padRight=1;uint8_tstrideH=1,strideW=1;uint16_tcoutBlocks,ho,wo,howo,howoRound;uint32_tfeatureMapA1Size,weightA1Size,featureMapA2Size,weightB2Size,dstSize,dstCO1Size;uint16_tm,k,n;uint8_tfmRepeat,weRepeat;};extern"C"__global____aicore__voidload_data_simple_kernel(__gm__uint8_t*fmGm,__gm__uint8_t*weGm,__gm__uint8_t*dstGm){KernelLoadDataop;op.Init(fmGm,weGm,dstGm);op.Process();} |
| --- | --- |
