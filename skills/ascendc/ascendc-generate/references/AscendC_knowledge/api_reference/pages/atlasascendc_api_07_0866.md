# Transpose Tiling-张量变换-高阶API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0866
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0866.html
---

# Transpose Tiling

#### 功能说明

用于获取Transpose Tiling参数。

#### 函数原型

![](../images/atlasascendc_api_07_10005_img_003.png)

- 获取最小临时空间大小1voidGetTransposeMaxMinTmpSize(constge::Shape&srcShape,constuint32_ttypeSize,constuint32_ttransposeTypeIn,uint32_t&maxValue,uint32_t&minValue)1voidGetConfusionTransposeMaxMinTmpSize(constge::Shape&srcShape,constuint32_ttypeSize,constuint32_ttransposeTypeIn,uint32_t&maxValue,uint32_t&minValue)
- 获取Transpose Tiling1voidGetTransposeTilingInfo(constge::Shape&srcShape,constuint32_tstackBufferSize,constuint32_ttypeSize,constuint32_ttransposeTypeIn,optiling::ConfusionTransposeTiling&tiling)1voidGetTransposeTilingInfo(constge::Shape&srcShape,constuint32_tstackBufferSize,constuint32_ttypeSize,constuint32_ttransposeTypeIn,AscendC::tiling::ConfusionTransposeTiling&tiling)1voidGetConfusionTransposeOnlyTilingInfo(constge::Shape&srcShape,constuint32_tstackBufferSize,constuint32_ttypeSize,optiling::ConfusionTransposeTiling&tiling)1voidGetConfusionTransposeOnlyTilingInfo(constge::Shape&srcShape,constuint32_tstackBufferSize,constuint32_ttypeSize,AscendC::tiling::ConfusionTransposeTiling&tiling)1voidGetConfusionTransposeTilingInfo(constge::Shape&srcShape,constuint32_tstackBufferSize,constuint32_ttypeSize,constuint32_ttransposeTypeIn,optiling::ConfusionTransposeTiling&tiling)1voidGetConfusionTransposeTilingInfo(constge::Shape&srcShape,constuint32_tstackBufferSize,constuint32_ttypeSize,constuint32_ttransposeTypeIn,AscendC::tiling::ConfusionTransposeTiling&tiling)

#### 参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入Tensor的shape信息，具体srcShape传入格式为：场景1：[B, N, S, H/N]场景2：[B, N, S, H/N]场景3：[B, N, S, H/N]场景4：[B, N, S, H/N]场景5：[B, N, S, H/N]场景6：[B, N, S, H/N]场景7：[H, W] |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| transposeTypeIn | 输入 | 选择数据排布及reshape的类型，根据输入数字选择对应的场景，参数范围为[1,7]。场景1（NZ2ND，1、2轴互换）：1场景2（NZ2NZ，1、2轴互换）：2场景3（NZ2NZ，尾轴切分）：3场景4（NZ2ND，尾轴切分）：4场景5（NZ2ND，尾轴合并）：5场景6（NZ2NZ，尾轴合并）：6场景7（二维转置）：7 |
| maxValue | 输出 | Transpose接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。说明：maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Transpose接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| srcShape | 输入 | 输入的shape信息，具体srcShape传入格式为：场景1：[B, N, S, H/N]场景2：[B, N, S, H/N]场景3：[B, N, S, H/N]场景4：[B, N, S, H/N]场景5：[B, N, S, H/N]场景6：[B, N, S, H/N]场景7：[H, W] |
| stackBufferSize | 输入 | 可供Transpose接口计算的空间大小，单位Byte。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| transposeTypeIn | 输入 | 选择数据排布及reshape的类型，根据输入数字选择对应的场景，参数范围为[1,7]。场景1（NZ2ND，1、2轴互换）：1场景2（NZ2NZ，1、2轴互换）：2场景3（NZ2NZ，尾轴切分）：3场景4（NZ2ND，尾轴切分）：4场景5（NZ2ND，尾轴合并）：5场景6（NZ2NZ，尾轴合并）：6场景7（二维转置）：7 |
| tiling | 输出 | 输入数据的切分信息。 |

#### 返回值说明

无

#### 约束说明

无

#### 调用示例

如下样例介绍了使用Transpose高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中为场景1，输入Tensor的shape大小为[1, 2, 64, 32]，输入的数据类型为half。

1. 将ConfusionTransposeTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。12345BEGIN_TILING_DATA_DEF(TilingData)// 注册一个tiling的类，以tiling的名字作为入参TILING_DATA_FIELD_DEF(uint32_t,tileNum);// 添加tiling字段，每个核上总计算数据分块个数...// 添加其他tiling字段TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling,confusionTransposeTilingData);// 将ConfusionTransposeTiling结构体参数增加至TilingData结构体END_TILING_DATA_DEF;
1. Tiling实现函数中，根据输入shape、可供计算的空间大小(stackBufferSize)等信息获取Transpose kernel侧接口所需tiling参数。12345678910111213141516171819202122232425262728namespaceoptiling{constuint32_tBLOCK_DIM=8;constuint32_tTILE_NUM=8;staticge::graphStatusTilingFunc(gert::TilingContext*context){TilingDatatiling;uint32_ttotalLength=context->GetInputTensor(0)->GetShapeSize();context->SetBlockDim(BLOCK_DIM);tiling.set_tileNum(TILE_NUM);// 设置其他Tiling参数...std::vector<int64_t>shapeVec={1,2,64,32};ge::ShapesrcShape(shapeVec);uint32_tmaxValue=0;uint32_tminValue=0;AscendC::GetTransposeMaxMinTmpSize(srcShape,sizeof(half),maxValue,minValue);// 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小constuint32_tstackBufferSize=minValue;// 获取Transpose Tiling参数AscendC::GetTransposeTilingInfo(srcShape,stackBufferSize,sizeof(half),1,tiling.confusionTransposeTilingData);...// 其他逻辑tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());context->SetTilingKey(1);returnge::GRAPH_SUCCESS;}}// namespace optiling
1. 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的ConfusionTransposeTiling信息传入Transpose接口参与计算。完整的kernel侧样例请参考Transpose。1234567extern"C"__global____aicore__voidfunc_custom(GM_ADDRsrc_gm,GM_ADDRdst_gm,GM_ADDRworkspace,GM_ADDRtiling){GET_TILING_DATA(TilingData,tiling);KernelTranspose<half>op;op.Init(src_gm,dst_gm,TilingData.confusionTransposeTilingData);op.Process();}
