# Transpose-数据转换-矢量计算-基础API-Ascend C算子开发接口-API-CANN社区版8.5.0开发文档-昇腾社区
**页面ID:** atlasascendc_api_07_0199
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0199.html
---

# Transpose

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

用于实现16*16的二维矩阵数据块转置或者[N,C,H,W]与[N,H,W,C]数据格式互相转换。

#### 函数原型

- 普通转置，支持16*16的二维矩阵数据块进行转置12template<typenameT>__aicore__inlinevoidTranspose(constLocalTensor<T>&dst,constLocalTensor<T>&src)

- 增强转置，支持16*16的二维矩阵数据块转置，支持[N,C,H,W]与[N,H,W,C]互相转换12template<typenameT>__aicore__inlinevoidTranspose(constLocalTensor<T>&dst,constLocalTensor<T>&src,constLocalTensor<uint8_t>&sharedTmpBuffer,constTransposeParamsExt&transposeParams)

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。普通转置:Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：uint16_t/int16_t/halfAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：uint16_t/int16_t/halfAtlas 200I/500 A2 推理产品，支持的数据类型为：uint16_t/int16_t/halfAtlas 推理系列产品AI Core，支持的数据类型为：uint16_t/int16_t/halfAtlas 训练系列产品，支持的数据类型为：uint16_t/int16_t/half增强转置:transposeType为TRANSPOSE_ND2ND_B16：Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：uint16_tAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：uint16_tAtlas 200I/500 A2 推理产品，支持的数据类型为：uint16_tAtlas 推理系列产品AI Core，支持的数据类型为：uint16_ttransposeType为TRANSPOSE_NCHW2NHWC或TRANSPOSE_NHWC2NCHW：Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/floatAtlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/floatAtlas 推理系列产品AI Core，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/float |

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。 |
| src | 输入 | 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与dst保持一致。 |
| sharedTmpBuffer | 输入 | 共享的临时Buffer，sharedTmpBuffer的大小参考表4。 |
| transposeParams | 输入 | 控制Transpose的数据结构。结构体内包含：输入的shape信息和transposeType参数。该数据结构的定义请参考表3。12345678910111213141516structTransposeParamsExt{__aicore__TransposeParamsExt(){}__aicore__TransposeParamsExt(constuint16_tnSizeIn,constuint16_tcSizeIn,constuint16_thSizeIn,constuint16_twSizeIn,constTransposeTypetransposeTypeIn):nSize(nSizeIn),cSize(cSizeIn),hSize(hSizeIn),wSize(wSizeIn),transposeType(transposeTypeIn){}uint16_tnSize=0;uint16_tcSize=0;uint16_thSize=0;uint16_twSize=0;TransposeTypetransposeType=TransposeType::TRANSPOSE_ND2ND_B16;}; | 12345678910111213141516 | structTransposeParamsExt{__aicore__TransposeParamsExt(){}__aicore__TransposeParamsExt(constuint16_tnSizeIn,constuint16_tcSizeIn,constuint16_thSizeIn,constuint16_twSizeIn,constTransposeTypetransposeTypeIn):nSize(nSizeIn),cSize(cSizeIn),hSize(hSizeIn),wSize(wSizeIn),transposeType(transposeTypeIn){}uint16_tnSize=0;uint16_tcSize=0;uint16_thSize=0;uint16_twSize=0;TransposeTypetransposeType=TransposeType::TRANSPOSE_ND2ND_B16;}; |
| 12345678910111213141516 | structTransposeParamsExt{__aicore__TransposeParamsExt(){}__aicore__TransposeParamsExt(constuint16_tnSizeIn,constuint16_tcSizeIn,constuint16_thSizeIn,constuint16_twSizeIn,constTransposeTypetransposeTypeIn):nSize(nSizeIn),cSize(cSizeIn),hSize(hSizeIn),wSize(wSizeIn),transposeType(transposeTypeIn){}uint16_tnSize=0;uint16_tcSize=0;uint16_thSize=0;uint16_twSize=0;TransposeTypetransposeType=TransposeType::TRANSPOSE_ND2ND_B16;}; |

| 参数名称 | 含义 |
| --- | --- |
| nSize | n轴长度。默认值为0。二维矩阵数据块转置，无需传入，传入数值无效。[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：nSize∈[0, 65535]。 |
| cSize | c轴长度。默认值为0。二维矩阵数据块转置，无需传入，传入数值无效。[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：cSize∈[0, 4095] |
| hSize | h轴长度。默认值为0。二维矩阵数据块转置，固定传入16。[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：hSize * wSize ∈[0, 4095]，hSize * wSize * sizeof(T)需要保证32B对齐。 |
| wSize | w轴长度。默认值为0。二维矩阵数据块转置，固定传入16。[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：hSize * wSize ∈[0, 4095]，hSize * wSize * sizeof(T)需要保证32B对齐。 |
| transposeType | 数据排布及reshape的类型，类型为TransposeType枚举类。默认值为TRANSPOSE_ND2ND_B16。123456789101112131415enumclassTransposeType:uint8_t{TRANSPOSE_TYPE_NONE,// API不做任何处理TRANSPOSE_NZ2ND_0213,// 当前不支持TRANSPOSE_NZ2NZ_0213,// 当前不支持TRANSPOSE_NZ2NZ_012_WITH_N,// 当前不支持TRANSPOSE_NZ2ND_012_WITH_N,// 当前不支持TRANSPOSE_NZ2ND_012_WITHOUT_N,// 当前不支持TRANSPOSE_NZ2NZ_012_WITHOUT_N,// 当前不支持TRANSPOSE_ND2ND_ONLY,// 当前不支持TRANSPOSE_ND_UB_GM,// 当前不支持TRANSPOSE_GRAD_ND_UB_GM,// 当前不支持TRANSPOSE_ND2ND_B16,// [16,16]二维矩阵转置TRANSPOSE_NCHW2NHWC,// [N,C,H,W]->[N,H,W,C]，TRANSPOSE_NHWC2NCHW// [N,H,W,C]->[N,C,H,W]}; | 123456789101112131415 | enumclassTransposeType:uint8_t{TRANSPOSE_TYPE_NONE,// API不做任何处理TRANSPOSE_NZ2ND_0213,// 当前不支持TRANSPOSE_NZ2NZ_0213,// 当前不支持TRANSPOSE_NZ2NZ_012_WITH_N,// 当前不支持TRANSPOSE_NZ2ND_012_WITH_N,// 当前不支持TRANSPOSE_NZ2ND_012_WITHOUT_N,// 当前不支持TRANSPOSE_NZ2NZ_012_WITHOUT_N,// 当前不支持TRANSPOSE_ND2ND_ONLY,// 当前不支持TRANSPOSE_ND_UB_GM,// 当前不支持TRANSPOSE_GRAD_ND_UB_GM,// 当前不支持TRANSPOSE_ND2ND_B16,// [16,16]二维矩阵转置TRANSPOSE_NCHW2NHWC,// [N,C,H,W]->[N,H,W,C]，TRANSPOSE_NHWC2NCHW// [N,H,W,C]->[N,C,H,W]}; |
| 123456789101112131415 | enumclassTransposeType:uint8_t{TRANSPOSE_TYPE_NONE,// API不做任何处理TRANSPOSE_NZ2ND_0213,// 当前不支持TRANSPOSE_NZ2NZ_0213,// 当前不支持TRANSPOSE_NZ2NZ_012_WITH_N,// 当前不支持TRANSPOSE_NZ2ND_012_WITH_N,// 当前不支持TRANSPOSE_NZ2ND_012_WITHOUT_N,// 当前不支持TRANSPOSE_NZ2NZ_012_WITHOUT_N,// 当前不支持TRANSPOSE_ND2ND_ONLY,// 当前不支持TRANSPOSE_ND_UB_GM,// 当前不支持TRANSPOSE_GRAD_ND_UB_GM,// 当前不支持TRANSPOSE_ND2ND_B16,// [16,16]二维矩阵转置TRANSPOSE_NCHW2NHWC,// [N,C,H,W]->[N,H,W,C]，TRANSPOSE_NHWC2NCHW// [N,H,W,C]->[N,C,H,W]}; |

| transposeType | sharedTmpBuffer所需的大小 |
| --- | --- |
| TRANSPOSE_ND2ND_B16 | 不需要临时Buffer。 |
| TRANSPOSE_NCHW2NHWC | 针对以下型号：Atlas 推理系列产品AI Core不需要临时Buffer。针对以下型号：Atlas A2 训练系列产品/Atlas A2 推理系列产品Atlas A3 训练系列产品/Atlas A3 推理系列产品临时Buffer的大小按照下述计算规则（伪代码）进行计算。123autoh0=16;// 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16autow0=32/sizeof(type);// type代表数据类型autotmpBufferSize=(cSize+2)*h0*w0*sizeof(type); | 123 | autoh0=16;// 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16autow0=32/sizeof(type);// type代表数据类型autotmpBufferSize=(cSize+2)*h0*w0*sizeof(type); |
| 123 | autoh0=16;// 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16autow0=32/sizeof(type);// type代表数据类型autotmpBufferSize=(cSize+2)*h0*w0*sizeof(type); |
| TRANSPOSE_NHWC2NCHW | 针对以下型号：Atlas 推理系列产品AI Core不需要临时Buffer。针对以下型号：Atlas A2 训练系列产品/Atlas A2 推理系列产品Atlas A3 训练系列产品/Atlas A3 推理系列产品临时Buffer的大小按照下述计算规则（伪代码）进行计算。123autoh0=16;// 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16autow0=32/sizeof(type);// type代表数据类型autotmpBufferSize=(cSize*2+1)*h0*w0*sizeof(type); | 123 | autoh0=16;// 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16autow0=32/sizeof(type);// type代表数据类型autotmpBufferSize=(cSize*2+1)*h0*w0*sizeof(type); |
| 123 | autoh0=16;// 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16autow0=32/sizeof(type);// type代表数据类型autotmpBufferSize=(cSize*2+1)*h0*w0*sizeof(type); |

#### 返回值说明

无

#### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 普通转置接口支持src和dst复用。
- 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。

#### 调用示例

- 普通接口调用示例，该示例对[16,16]的half类型矩阵进行转置。12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455565758#include"kernel_operator.h"classKernelTranspose{public:__aicore__inlineKernelTranspose(){}__aicore__inlinevoidInit(__gm__uint8_t*src,__gm__uint8_t*dstGm){srcGlobal.SetGlobalBuffer((__gm__half*)src);dstGlobal.SetGlobalBuffer((__gm__half*)dstGm);pipe.InitBuffer(inQueueSrc,1,srcDataSize*sizeof(half));pipe.InitBuffer(outQueueDst,1,dstDataSize*sizeof(half));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.AllocTensor<half>();AscendC::DataCopy(srcLocal,srcGlobal,srcDataSize);inQueueSrc.EnQue(srcLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<half>srcLocal=inQueueSrc.DeQue<half>();AscendC::LocalTensor<half>dstLocal=outQueueDst.AllocTensor<half>();AscendC::Transpose<half>(dstLocal,srcLocal);outQueueDst.EnQue<half>(dstLocal);inQueueSrc.FreeTensor(srcLocal);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<half>dstLocal=outQueueDst.DeQue<half>();AscendC::DataCopy(dstGlobal,dstLocal,dstDataSize);outQueueDst.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrc;AscendC::TQue<AscendC::TPosition::VECOUT,1>outQueueDst;AscendC::GlobalTensor<half>srcGlobal,dstGlobal;intsrcDataSize=256;intdstDataSize=256;};extern"C"__global____aicore__voidtranspose_kernel(__gm__uint8_t*src,__gm__uint8_t*dstGm){KernelTransposeop;op.Init(src,dstGm);op.Process();}输入数据src_gm:
[[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.
   14.  15.]
 [ 16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.  29.
   30.  31.]
 [ 32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  42.  43.  44.  45.
   46.  47.]
 [ 48.  49.  50.  51.  52.  53.  54.  55.  56.  57.  58.  59.  60.  61.
   62.  63.]
 [ 64.  65.  66.  67.  68.  69.  70.  71.  72.  73.  74.  75.  76.  77.
   78.  79.]
 [ 80.  81.  82.  83.  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.
   94.  95.]
 [ 96.  97.  98.  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109.
  110. 111.]
 [112. 113. 114. 115. 116. 117. 118. 119. 120. 121. 122. 123. 124. 125.
  126. 127.]
 [128. 129. 130. 131. 132. 133. 134. 135. 136. 137. 138. 139. 140. 141.
  142. 143.]
 [144. 145. 146. 147. 148. 149. 150. 151. 152. 153. 154. 155. 156. 157.
  158. 159.]
 [160. 161. 162. 163. 164. 165. 166. 167. 168. 169. 170. 171. 172. 173.
  174. 175.]
 [176. 177. 178. 179. 180. 181. 182. 183. 184. 185. 186. 187. 188. 189.
  190. 191.]
 [192. 193. 194. 195. 196. 197. 198. 199. 200. 201. 202. 203. 204. 205.
  206. 207.]
 [208. 209. 210. 211. 212. 213. 214. 215. 216. 217. 218. 219. 220. 221.
  222. 223.]
 [224. 225. 226. 227. 228. 229. 230. 231. 232. 233. 234. 235. 236. 237.
  238. 239.]
 [240. 241. 242. 243. 244. 245. 246. 247. 248. 249. 250. 251. 252. 253.
  254. 255.]]

输出数据dst_gm:
[[  0.  16.  32.  48.  64.  80.  96. 112. 128. 144. 160. 176. 192. 208.
  224. 240.]
 [  1.  17.  33.  49.  65.  81.  97. 113. 129. 145. 161. 177. 193. 209.
  225. 241.]
 [  2.  18.  34.  50.  66.  82.  98. 114. 130. 146. 162. 178. 194. 210.
  226. 242.]
 [  3.  19.  35.  51.  67.  83.  99. 115. 131. 147. 163. 179. 195. 211.
  227. 243.]
 [  4.  20.  36.  52.  68.  84. 100. 116. 132. 148. 164. 180. 196. 212.
  228. 244.]
 [  5.  21.  37.  53.  69.  85. 101. 117. 133. 149. 165. 181. 197. 213.
  229. 245.]
 [  6.  22.  38.  54.  70.  86. 102. 118. 134. 150. 166. 182. 198. 214.
  230. 246.]
 [  7.  23.  39.  55.  71.  87. 103. 119. 135. 151. 167. 183. 199. 215.
  231. 247.]
 [  8.  24.  40.  56.  72.  88. 104. 120. 136. 152. 168. 184. 200. 216.
  232. 248.]
 [  9.  25.  41.  57.  73.  89. 105. 121. 137. 153. 169. 185. 201. 217.
  233. 249.]
 [ 10.  26.  42.  58.  74.  90. 106. 122. 138. 154. 170. 186. 202. 218.
  234. 250.]
 [ 11.  27.  43.  59.  75.  91. 107. 123. 139. 155. 171. 187. 203. 219.
  235. 251.]
 [ 12.  28.  44.  60.  76.  92. 108. 124. 140. 156. 172. 188. 204. 220.
  236. 252.]
 [ 13.  29.  45.  61.  77.  93. 109. 125. 141. 157. 173. 189. 205. 221.
  237. 253.]
 [ 14.  30.  46.  62.  78.  94. 110. 126. 142. 158. 174. 190. 206. 222.
  238. 254.]
 [ 15.  31.  47.  63.  79.  95. 111. 127. 143. 159. 175. 191. 207. 223.
  239. 255.]]

- 增强接口调用示例，完成half类型的[N,C,H,W]->[N,H,W,C]转置。1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950515253545556575859606162636465666768697071727374#include"kernel_operator.h"template<typenameT>classKernel4dTrans{public:__aicore__inlineKernel4dTrans(){}__aicore__inlinevoidInit(__gm__uint8_t*srcGm,__gm__uint8_t*dstGm){inputSize=N*C*H*W;tmpBufferSize=(C+2)*16*16;srcGlobal.SetGlobalBuffer((__gm__T*)srcGm);dstGlobal.SetGlobalBuffer((__gm__T*)dstGm);pipe.InitBuffer(inQueueSrcVecIn,1,inputSize*sizeof(T));pipe.InitBuffer(inQueueSrcVecOut,1,inputSize*sizeof(T));pipe.InitBuffer(tmpQueue,1,tmpBufferSize*sizeof(T));}__aicore__inlinevoidProcess(){CopyIn();Compute();CopyOut();}private:__aicore__inlinevoidCopyIn(){AscendC::LocalTensor<T>srcLocal=inQueueSrcVecIn.AllocTensor<T>();AscendC::DataCopy(srcLocal,srcGlobal,inputSize);inQueueSrcVecIn.EnQue(srcLocal);}__aicore__inlinevoidCompute(){AscendC::LocalTensor<T>srcLocal=inQueueSrcVecIn.DeQue<T>();AscendC::LocalTensor<T>dstLocal=inQueueSrcVecOut.AllocTensor<T>();AscendC::LocalTensor<uint8_t>stackBuffer=tmpQueue.AllocTensor<uint8_t>();AscendC::TransposeParamsExttransposeParams;transposeParams.nSize=N;transposeParams.cSize=C;transposeParams.hSize=H;transposeParams.wSize=W;transposeParams.transposeType=transposeType;AscendC::Transpose(dstLocal,srcLocal,stackBuffer,transposeParams);inQueueSrcVecOut.EnQue<T>(dstLocal);inQueueSrcVecIn.FreeTensor(srcLocal);tmpQueue.FreeTensor(stackBuffer);}__aicore__inlinevoidCopyOut(){AscendC::LocalTensor<T>dstLocal=inQueueSrcVecOut.DeQue<T>();AscendC::DataCopy(dstGlobal,dstLocal,inputSize);inQueueSrcVecOut.FreeTensor(dstLocal);}private:AscendC::TPipepipe;AscendC::TQue<AscendC::TPosition::VECIN,1>inQueueSrcVecIn;AscendC::TQue<AscendC::TPosition::VECOUT,1>inQueueSrcVecOut;AscendC::TQue<AscendC::TPosition::VECCALC,1>tmpQueue;AscendC::GlobalTensor<T>srcGlobal;AscendC::GlobalTensor<T>dstGlobal;uint32_tN=3;uint32_tC=3;uint32_tH=2;uint32_tW=8;uint32_tinputSize,tmpBufferSize;AscendC::TransposeTypetransposeType=AscendC::TransposeType::TRANSPOSE_NCHW2NHWC;};extern"C"__global____aicore__voidtranspose_kernel(__gm__uint8_t*srcGm,__gm__uint8_t*dstGm){Kernel4dTrans<half>op;op.Init(srcGm,dstGm);op.Process();}
