---
name: ascendc-registry-invoke-to-direct-invoke
description: 当用户想把自定义算子工程中的 kernel 模板改造成 `<<<>>>` kernel 直调形式，或从自定义算子工程中抽取某个 kernel 模板并转换成 `<<<>>>` 直调方式时使用。触发：用户提到"自定义算子转直调"、"从算子工程抽 kernel"、"kernel 模板改 `<<<>>>`"等。不适用于从零开发新算子
---

# AscendC 自定义算子转 `<<<>>>` kernel 直调改造

这个 skill 处理的是两类任务：
1. **把已有 AscendC 自定义算子改造成 `<<<>>>` kernel 直调形态**
2. **从某个现有模板或算子实现中抽取 kernel/tiling/host 关键部分，并转换成 `<<<>>>` 方式**

目标不是“把文件复制过来”，而是识别并拆解现有实现里的 kernel、tiling、host glue 与外部依赖，把它整理成一个 **本地可维护、依赖闭环清晰、kernel 语义不变、可直接 launch** 的版本。

## 默认处理范围

它默认解决的是：
- 从 `op_kernel/archXX` 或类似源树中抽出 header-only / inline kernel 代码
- 删除 `#include "..` 开头的跨目录依赖
- 把真正需要的函数 / 常量 / traits / helper 收口到本地
- 在保持 kernel 计算语义不变的前提下，补齐现有算子转 `<<<>>>` kernel 直调所需的最小必要改写

它默认 **不处理**：
- `aclnn` / `OpDef` / 注册流程
- 完整的新算子设计与开发

它可以处理两类任务：
1. **kernel-only 去耦 / 自包含**
2. **kernel + tiling + `<<<>>>` 直调入口 / standalone sample host glue 适配**

如果用户已经明确告诉你：
- kernel 代码在哪里
- tiling 代码在哪里

就不要再拆成两个技能链路，直接按同一条自定义算子转直调工作流推进。

---

## 先判断任务属于哪一类

优先把任务识别为以下一种：

1. **kernel 本地化搬运**
   - 例如：把 `arch35/` 目录搬到当前路径，删掉相对 include，并把依赖函数搬到本地
2. **kernel 头文件去耦**
   - 例如：不再依赖 `../../rms_norm/...`、`../inc/platform.h`
3. **最小依赖闭包提取**
   - 例如：只搬 `DataCopyImpl` / `ComputeRstd` / `CeilDiv`，不整包复制上游头文件
4. **kernel 入口本地落地**
   - 例如：把 `*_apt.cpp` 一起放到目标目录，并显式指出还缺哪些外部宏/tiling 结构

如果用户明确说：
- “适配到 cann-samples”
- “做 standalone story”
- “要能独立编译运行的 story 工程”

则不要误判成“只做 kernel 去耦”；要按 **kernel + tiling + standalone sample/story host glue** 的完整链路处理。

---

## 默认工作流

> **信息来源约束**：执行本工作流时，只读两类代码：
> 1. **原始算子源码**（用户提供的 kernel 源目录 + tiling 源目录）
> 2. **本 skill 内嵌的模板和规则**
>
> **不要读目标仓库（如 cann-samples）中的其他 sample、story、utils 或 cpp 文件来"参考写法"。** 所有 host 样板、CMakeLists.txt 结构、类型转换 trait、ACL 初始化模式等，均已内嵌在本 skill 的"Host 驱动代码模板"章节中。

### Step 1: 先确认交付边界，再决定走 kernel-only 还是 `<<<>>>` 直调链路
先确认这 3 件事：
- **源目录**：通常是 `op_kernel/` 或 `op_kernel/archXX/`
- **目标目录**：当前路径根、子目录，还是保留源层级
- **交付边界**：只是 kernel 自包含，还是还要让 `*_apt.cpp` / tiling / `<<<>>>` 入口一并可独立编译调用

默认假设：
- 先做 **kernel 代码适配**
- 不主动扩展到 host / graph / 注册文件
- 不重写算法逻辑，只做依赖适配

但如果用户明确要求 standalone story / cann-samples / `<<<>>>` 直调：
- 继续保持 **kernel 语义不变**
- 同时把 tiling / host glue / 直调入口补齐到能在目标工程里独立维护
- 仍然不要主动扩展到 `aclnn` / `OpDef` / 注册链路

### Step 2: 盘点源文件集合
用 `Glob` / `Read` / `Grep` 先确认：
- 目录下有哪些 `.h` / `.cpp`
- 哪个文件是入口文件（例如 `*_apt.cpp`）
- 哪些是 `archXX` 特化头
- 哪些文件已经是“局部本地化后的二次封装”

特别注意：
- 很多 AscendC kernel 是 **header-only** 风格
- 真正要搬运的实现，往往藏在 `common` / `regbase_common` / `base` 头里

### Step 3: 只追踪 `..` 相对 include，并建立“符号级依赖图”
必须先找出所有：
- `#include "../..."`
- `#include "../../..."`

然后对每个相对 include 做两件事：
1. 记录 **当前 kernel 文件真正使用了哪些符号**
2. 记录这些符号的 **真实定义位置**

不要因为某个头被 include 了，就整包复制整个头文件。

输出时至少要列出：
- 相对 include 路径
- 被实际使用的函数 / 常量 / traits / 类型 / 宏
- 是否存在“表面来自 A，实际定义在 B”的转手依赖

### Step 4: 优先选“保留原 kernel 文件 + 新增 local helper 头”的收口方式
默认推荐结构：
- 保留原始 kernel 文件主体
- 新增一个同目录 helper，例如：`xxx_local_deps.h`
- 把外部依赖的最小闭包集中搬到 helper 里

只有在依赖非常少时，才考虑把外部内容直接塞回原头文件。

推荐结构通常是：
- `foo_regbase.h`
- `foo_regbase_common.h`
- `foo_regbase_split_d.h`
- `foo_local_deps.h`
- 可选：`foo_apt.cpp` 或 `foo_apt.h`

这样做的好处：
- 原有 kernel 结构不散
- 外部依赖边界清楚
- 后续复查时容易区分“本地逻辑”和“搬运依赖”

### Step 5: 只搬“最小依赖闭包”
把依赖分成 4 类来搬。

#### 5.1 平台常量
来自 `platform` 命名空间或平台头文件的常量/函数（如 `GetUbBlockSize`、`GetVRegSize`）。

如果只是返回固定常量，优先改成：
- 小型 `constexpr` 包装
- 或直接在本地 helper 中保留最小 `platform` namespace 子集

#### 5.2 基础常量 / traits
来自 `*_base.h` 或公共头的基础定义，如 buffer 数量常量、对齐/取整工具函数（`CeilDiv` 等）、类型 traits（`is_same`、`bfloat16_t` 兼容定义等）。

这类通常来自某个 `*_base.h`，但真正需要的往往只是一小段，不要整包搬。

#### 5.3 common 层工具
来自公共 common/reduce 层头文件的工具函数，如对齐函数（`CeilAlign`）、cast 辅助函数、多级规约函数等。

这类最容易出现”命名空间错位”。必须核实符号真实归属，而不是照抄 `using`。

#### 5.4 算法 helper / regbase helper
当前算子特有的计算辅助函数，如数据搬运 helper、规约计算函数、数学工具函数等。

只复制 **当前 kernel 真正调用到的函数**，并补齐它们的直接闭包。
不要把整个上游 `*_common.h` / `*_regbase_common.h` 原样拖进目标目录。

### Step 6: 如果目标是 `<<<>>>` 直调 / standalone story，再补 tiling / host glue，但保持 kernel 独立
当用户明确要 cann-samples / standalone story / `<<<>>>` 直调时：
- 优先把 tiling struct 改成当前目录可见的 plain struct，或本地等价结构
- 把 `op_host` 的 tiling 逻辑改成本地 helper，而不是继续绑定原工程注册框架
- 让入口按当前 story 的 host dispatch 或 `<<<>>>` launch 方式工作，而不是继续依赖原工程分发宏
- 保持 kernel 实现头尽量独立，host 逻辑只做最小串联

这里的重点是：
- **可以本地化 tiling 与 host glue**
- **要把现有算子形态改造成可直接 launch 的入口**
- **不要把任务扩展成正式 op 注册链路改造**

---

## 决策规则

### 规则 1：优先复用当前 kernel 目录里已经存在的“半本地化”文件
如果源目录中已经有一个 `*_common.h` 明显是从上游算法模板改出来的：
- 优先复用它
- 只把缺失依赖补齐
- 不要回退去重建另一套结构

### 规则 2：对”表面归属”和”真实定义”保持怀疑
典型坑：
- `using A::someHelper;` — 但 `someHelper` 的真实定义其实在命名空间 `B` 中，`A` 只是通过 include 间接暴露
- 例如：`using RmsNorm::castTraitB162B32;` 但真实定义在 `NormCommon`

处理原则：
- 以真实定义源为准
- 迁移时顺手修正 `using` 归属
- 不保留错误的中转命名空间引用

### 规则 3：删除死引用，而不是机械保留
如果某个 `using` 或 include 只是在源码里“看起来像要用”，但实际未被引用：
- 直接删
- 不做兼容性保留

典型模式：
- `using XXX::SomeHelper;` 只声明不使用（如原工程某个算子的 `DataCopyCustom`、`ReduceGrad` 等）

### 规则 4：kernel 本地化完成，不等于入口文件可独立编译或直接 launch
如果用户只要求 kernel 本地化，做到以下即可视为完成：
- kernel 头文件不再依赖 `..` 相对 include
- helper 头已收口最小依赖闭包

但如果入口文件 `*_apt.cpp` 还依赖：
- `DTYPE_X1`
- `GET_TILING_DATA_WITH_STRUCT`
- `TILING_KEY_IS`
- 外部 tiling struct

必须在结果里明确说明：
- **kernel 已本地化**
- **入口仍依赖外部编译环境**
- 若要独立编译或直接 launch，需要继续本地化 tiling / dispatch / dtype 宏

### 规则 5：如果用户要“可独立编译”，优先本地化 tiling struct 或改 host dispatch
常见做法：
- 把 tiling struct 改成 plain struct
- 本地补 `GET_TILING_DATA_WITH_STRUCT` 依赖
- 或改成 Host dispatch / by-value tiling，避免继续绑定原工程宏

### 规则 6：如果 kernel 入口参数保持 `GM_ADDR`，host 侧 device 指针也从一开始就保持 `GM_ADDR`
常见做法：
- host 侧直接声明：`GM_ADDR inputDevice = nullptr;`
- 用 `aclrtMalloc(reinterpret_cast<void**>(&inputDevice), size, ...)` 申请
- kernel launch 时直接传 `inputDevice`
- 用 `aclrtFree(inputDevice)` 释放

不要这样做：
- 先把 device 指针存成 `void*`
- 或包在 `std::unique_ptr<void>` 里
- 然后在 launch 点写 `reinterpret_cast<GM_ADDR>(ptr)`

原因：
- 这不是“风格问题”，而是 **AscendC 编译器会直接拒绝从 `void*` 到 `GM_ADDR` 的这种转换**
- 一旦入口 ABI 决定保留 `GM_ADDR`，host 侧也要沿着这条 ABI 一致到底

### 规则 8：TilingData 替换 `BEGIN_TILING_DATA_DEF` 时只需转成普通 C++ struct
把 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` 宏机械替换成等价的 plain C++ struct 即可。不需要额外加 `#pragma pack(push, 1)`。

只保留 kernel 实际访问的字段。原 tiling struct 里给其他 variant 用的字段（如 `activateLeft`、`biasIsEmpty`）直接删除。

### 规则 9：`<<<>>>` 入口函数是否加 `KERNEL_TASK_TYPE_DEFAULT` 取决于原始代码
如果原始 `*_apt.cpp` 中已有 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 等宏，适配时保留。如果原始代码没有，不要主动添加。这只是保持原样，不是 `<<<>>>` 直调的必要条件。

### 规则 10：按 tilingKey 拆独立入口函数，而不是运行时 `TILING_KEY_IS` 分发
原工程用单一 `extern “C”` 入口 + `TILING_KEY_IS(N)` 做运行时分发。tilingKey 的划分维度因算子而异——有时按 dtype（如 swi_glu 按 fp32/fp16/bf16），有时按计算模式（如 fullload/splitd），也可能按其他规则。`<<<>>>` 直调时改为：
- 按 tilingKey 的实际划分维度拆成独立的 `__global__` 函数（如 `xxx_fp32()`、`xxx_fp16()` 或 `xxx_fullload()`、`xxx_splitd()`）
- 各自只实例化对应的模板路径
- Host 侧根据实际条件选择调用哪个函数
- `extern “C”` 去掉
- `GET_TILING_DATA(xxx, tiling)` 去掉，tiling 改成 `const XxxTilingData tilingData` by-value 传入

### 规则 11：`local_deps.h` 即使无外部搬运依赖也要创建
即使 arch35 kernel 头没有 `..` 相对 include，也要创建 `xxx_local_deps.h`，用来集中 SDK 相关 include：
- `kernel_operator.h`（必须）
- `simt_api/asc_bf16.h`（如果用到 `bfloat16_t`）
- 其他按需添加

这样做的好处是：kernel 实现头只 include 一个统一入口，后续增减 SDK 依赖时只改一处。

### 规则 12：host tiling 用 `PlatformAscendCManager::GetInstance()` 替代 `gert::TilingContext`
原工程 tiling 通过 `gert::TilingContext* context` 获取平台信息。独立工程中改为：
```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
const NpuArch npuArch = ascendcPlatform->GetCurNpuArch();
uint64_t ubSize = 0;
ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
const uint32_t totalCore = ascendcPlatform->GetCoreNumAiv();
```

需要 include `”platform/platform_ascendc.h”`。

同时去掉：
- `OP_LOGE` / `OP_LOGI` / `OP_LOGD` → 改为 `throw std::runtime_error(...)` 或 `printf`
- `ge::DataType` → 自定义 enum
- `gert::Shape` → 直接传 `uint64_t rowLen, colLen`
- `IMPL_OP_OPTILING` / `REGISTER_TILING_DATA_CLASS` → 直接删

### 规则 13：host golden 计算优先用 C++ 内联，不依赖 Python
standalone 工程的数据验证优先在 C++ 中完成：
- 用确定性模式生成输入（如线性递推 + 取模），避免依赖 numpy/random seed 对齐
- golden 计算用 float 精度完成后转回目标类型

### 规则 14：目标目录结构推荐 `include/kernel/` + `include/tiling/` + `src/` 分离
```
xxx_story/
  include/
    kernel/          # kernel 实现头 + local_deps.h + apt.h
    tiling/          # SwiGluTilingData plain struct + host tiling calculator
  src/
    xxx_story.cpp    # host 驱动 (acl init, malloc, launch, verify)
  CMakeLists.txt
```

CMakeLists.txt 通过 `target_include_directories` 设置 include 搜索路径：
```cmake
target_include_directories(xxx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tiling
)
```

### 规则 15：区分"原始代码逻辑"和"适配改造部分"
适配过程中必须清楚哪些代码是原始算子逻辑（计算、tiling 数学、数据搬运），哪些是为了脱离原工程框架而做的改造（入口拆分、宏替换、平台接口替换）。两者不要混在一起改：
- **原始逻辑**：保真搬运，不改语义，不"优化"
- **适配部分**：入口函数签名、TilingData struct 格式、平台 API 调用方式、框架宏替换

如果不确定某段代码属于哪一类，先看原始代码确认。

### 规则 16：适配结果必须自包含，不依赖目标仓库中的其他代码
生成的 standalone story / `<<<>>>` 直调工程必须只依赖：
- CANN SDK 头文件（`kernel_operator.h`、`platform_ascendc.h` 等）
- ACL 运行时库（`aclrt*` 系列）
- 标准 C++ 库

不能依赖目标仓库（如 cann-samples）中的其他 sample、公共 utils、或共享头文件。所有需要的代码必须在 story 目录内闭环。

### 规则 17：使用 MicroAPI/Reg 编程模型的 kernel 实现头必须用 `#if !defined(__NPU_HOST__)` 保护

arch35 regbase kernel 大量使用 `AscendC::MicroAPI::RegTensor`、`MaskReg`、`CastTrait`、`DivSpecificMode` 等类型。这些类型仅在 device pass（`__NPU_ARCH__` 已定义）可用；host pass（`__NPU_HOST__ == 1`）不提供。

如果不加保护，host pass 编译会报 `no template named 'RegTensor' in namespace 'AscendC::Reg'` 等大量错误。

正确做法：
- 所有使用 MicroAPI/Reg 的 kernel 实现头（`*_regbase_*.h`、`*_base.h`），在 `#include` 之后、namespace 开始之前加 `#if !defined(__NPU_HOST__)`，在文件末尾 namespace 关闭后加对应的 `#endif`
- **不影响** `local_deps.h`、`tilingdata.h` 等纯类型定义头——它们在 host/device 两侧都需要可见

判断依据：grep 文件中是否出现 `MicroAPI::`、`Reg::`、`__VEC_SCOPE__`、`RegTensor`、`MaskReg`。如果有，就需要 `__NPU_HOST__` guard。

### 规则 18：`__global__` 入口函数用”body 内 `#if`”模式，不用”声明 + 定义分离”模式

`-xasc` 编译器对 host 和 device pass 分别编译同一个 `.cpp` 文件。`__global__` 函数在 host pass 需要可见（生成 `<<<>>>` launch stub），在 device pass 需要有完整实现。

**错误做法 1**：把 `__global__` 函数整体放在 `#if !defined(__NPU_HOST__)` 里
- 结果：host pass 看不到函数声明 → `use of undeclared identifier` 错误

**错误做法 2**：在 guard 外放 forward declaration，guard 内放定义
- 结果：device pass 同时看到声明和定义 → `call is ambiguous` 错误

**正确做法**：函数签名放在 guard 外，函数体内部用 `#if !defined(__NPU_HOST__)` 保护实现细节：
```cpp
__global__ __aicore__ void my_kernel(GM_ADDR x, ..., const TilingData td, uint32_t param)
{
#if !defined(__NPU_HOST__)
    // 完整 kernel 实现（使用 TPipe、MicroAPI 等 device-only 类型）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MyKernelClass<...> op(&pipe);
    op.Init(...);
    op.Process();
#endif
}
```
host pass 编译出空函数体，compiler 据此生成 launch stub；device pass 编译完整实现。

### 规则 19：`__global__` 入口不使用模板，改用显式非模板函数

`-xasc` 编译器在处理模板 `__global__` 函数 + `<<<>>>` 调用时，host pass 可能出现参数类型 mangling 错误。典型表现：
- 模板参数为 `bfloat16_t` 时，`GM_ADDR` 类型的参数被错误推断为 `__bf16`
- 链接时报 `undefined symbol`，符号签名中 `GM_ADDR` 参数变成了模板类型

**错误做法**：
```cpp
template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__global__ __aicore__ void dq_common_fullload(GM_ADDR x, ...);
// 然后 host 调用:
dq_common_fullload<bfloat16_t, int8_t, false, true><<<...>>>(xDev, ...);
```

**正确做法**：按 (xDtype, yDtype, hasSmooth, isSymmetrical) 组合创建显式命名的非模板函数：
```cpp
__global__ __aicore__ void dq_fullload_bf16_int8_nosmooth_sym(GM_ADDR x, ..., uint32_t useDb);
```
模板参数硬编码在函数名和函数体中。host 直接调用具名函数，无模板推断。

命名规范：`dq_{mode}_{xname}_{yname}_{smooth}_{sym}`

### 规则 7：抽取 tiling 数学逻辑时，优先逐分支保真，不要擅自把不同分支”统一写法”
尤其注意：
- `is32BAligned == 1` 的路径
- `is32BAligned == 0` 的路径
- 分母到底是原始 `baseColLen`，还是 `AlignUp(baseColLen, ubMinBlockLen)`

典型例子：
- 32B 对齐时：`baseRowLen = maxTileLen / baseColLen`
- 非 32B 对齐时：`baseRowLen = maxTileLen / AlignUp(baseColLen, ubMinBlockLen)`

不要为了“代码更整齐”把两条公式都改成后一种。

原因：
- 这会改变 tiling 结果
- 轻则性能漂移，重则和原始切块语义不一致

---

## 输出格式要求

完成这类任务时，结果里至少要给出：

1. **目标文件集合**
   - 最终落地了哪些 `.h` / `.cpp`
2. **被删除的相对 include**
   - 精确到文件和 include 语句
3. **本地化的依赖清单**
   - 按来源头文件列出搬了哪些符号
4. **helper 头职责**
   - 为什么新增它、里面装了哪类依赖
5. **剩余外部假设**
   - 哪些宏 / tiling / dtype 仍依赖外部环境
6. **静态验证结论**
   - 是否已没有 `..` include
   - 是否还有错误命名空间 `using`

如果任务明确是 `<<<>>>` 直调 / standalone story / cann-samples，结果中还应补一句：
- 当前版本是 **kernel-only 自包含**，还是已经做到 **kernel + tiling + direct-launch glue 闭环**

---

## 验证清单

快速执行时，直接按 `references/custom-op-to-kernel-launch-checklist.md` 做静态核验。

注意：
- 这个清单对 **自定义算子转直调过程中的 kernel 依赖清理** 最有用
- 如果任务是 standalone story / `<<<>>>` 直调，它仍可用于检查 kernel 闭包是否干净，但不能替代你对 host tiling / direct-launch glue 的单独判断

至少做以下静态检查：

### 1. 相对 include 清理
在目标文件集合中搜索：
- `#include "../`
- `#include "../../`

期望：无匹配。

### 2. 命名空间归属检查
搜索所有 `using XXX::symbol` 语句，逐条核实符号的真实定义位置是否和 `using` 声明的命名空间一致。

典型问题模式：`using A::foo`，但 `foo` 的真实定义在命名空间 `B` 中，`A` 只是通过 include 间接暴露了它。
如果发现归属不一致，必须修正为真实定义源。

### 3. 本地闭包检查
确认 kernel 代码中所有非 SDK 符号（函数、常量、traits、类型别名）要么来自本地 helper 头，要么来自稳定 SDK 头。

方法：在目标目录中 grep kernel 头文件引用的每个非 SDK 函数/常量，确认都能在本地解析。常见需要检查的类别：
- 工具函数（如 `CeilDiv`、`CeilAlign`、各种 `Compute*` helper）
- 平台常量（如 `GetUbBlockSize`、`GetVRegSize`）
- 类型 traits（如 cast 辅助函数、`is_same`）
- 业务常量（如 `BUFFER_NUM`、`V_LENGTH`）

### 4. 死引用检查
确认并删除：
- 未使用的 `using`
- 未使用的 include
- 仅为历史兼容留下的空壳依赖

### 5. 入口文件假设检查
如果同时适配了 `*_apt.cpp` 或 `*_apt.h`，必须明确检查：
- `DTYPE_X1`
- `GET_TILING_DATA_WITH_STRUCT`
- `TILING_KEY_IS`
- tiling struct 是否本地可见

### 6. Host launch ABI 一致性检查
如果最终 direct-launch 入口参数仍然是 `GM_ADDR`，必须明确检查：
- host 侧 device buffer 变量是不是也声明成 `GM_ADDR`
- `aclrtMalloc` 是否按 `reinterpret_cast<void**>(&devicePtr)` 形式写入这个 `GM_ADDR` 变量
- kernel launch 时是否直接传 `devicePtr`
- 是否还残留 `reinterpret_cast<GM_ADDR>(voidPtr)` 这种调用点强转

期望：
- 没有 `void* -> GM_ADDR` 的临门一脚强转
- host / launch / kernel 三侧 ABI 一致

### 7. Tiling 公式逐分支等价性检查
如果把 host tiling 从原工程提取成本地 helper，必须明确检查：
- 对齐分支和非对齐分支是否仍然分别保留
- 是否把原本分支不同的分母、上取整逻辑、上界逻辑误合并成同一套写法
- 关键公式是否逐条和上游比对过，而不是“凭感觉等价”

重点关注：
- `baseRowLen`
- `baseColLen`
- `tileLength`
- `is32BAligned` 相关分支

---

## 常见坑

### 1. 误把整份上游 `*_base.h` / `*_common.h` 原样复制
这会把大量无关实现也拖进来，后续更难维护。默认只搬最小闭包。

### 2. 不核实真实定义源
很多符号是通过中间头间接暴露的（如 `using A::foo` 但 `foo` 实际定义在另一个命名空间）。迁移时必须追到真实定义位置。

### 3. 只删 include，不补 helper
这样会让当前目录暂时“看起来干净”，但实际符号解析断掉。

### 4. 把纯 kernel 本地化和 host/工程集成混在一起
用户如果只要 kernel 代码适配，就不要主动扩展到 host / graph / 注册。

### 5. 忽略入口文件仍依赖原工程宏
`*_apt.cpp` 很容易在 include 改完之后，看起来也“在当前目录里了”，但其实仍不能独立编译。必须显式说明。

### 6. host 侧先用 `void*` 持有 device 指针，launch 时再强转成 `GM_ADDR`
这在普通 C++ 里看起来像小问题，但在 AscendC direct-launch 场景里经常会 **直接编译失败**。

正确做法是：
- 如果 kernel 入口参数是 `GM_ADDR`
- 那 host 侧申请 device 内存的变量也从一开始就用 `GM_ADDR`
- `aclrtMalloc(reinterpret_cast<void**>(&devicePtr), ...)`
- launch 直接传 `devicePtr`

### 8. 不区分 host pass 和 device pass 就直接搬运 regbase kernel 头
`-xasc` 编译器对同一个 `.cpp` 做两次编译（host pass 定义 `__NPU_HOST__`，device pass 定义 `__NPU_ARCH__`）。arch35 regbase kernel 使用的 `MicroAPI::RegTensor`、`MaskReg`、`CastTrait` 等类型 **仅 device pass 可用**。

如果把 kernel 实现头原样搬到 standalone 工程，host pass 会报几十个 `no template named 'RegTensor'` 错误。

修复方法：所有 regbase 实现头加 `#if !defined(__NPU_HOST__)` guard；`__global__` 入口用”body 内 guard”模式（见规则 18）。

### 9. `__global__` 入口用模板函数 + `<<<>>>` 调用导致 bf16 参数类型 mangling 错误
模板 `__global__` 函数在 `-xasc` host pass 处理 `<<<>>>` 语法时，当模板参数为 `bfloat16_t` 时，`GM_ADDR`（即 `__gm__ uint8_t*`）参数会被错误 mangle 为 `__bf16`。链接时报 `undefined symbol`，符号签名里出现 `__bf16` 而非 `unsigned char*`。

修复方法：改用非模板 `__global__` 函数（见规则 19），避免模板参数干扰 host pass 类型推断。

### 10. host 侧用自定义 struct（如 `SampleBFloat16`）作为 kernel 模板参数
host 定义的 `SampleBFloat16{uint16_t bits}` 仅用于 host 侧数据生成和 golden 计算。如果直接用作 kernel 模板参数（`LaunchKernel<SampleBFloat16>` → 实例化 `DynamicQuantRegbaseFullLoad<SampleBFloat16, ...>`），会导致 `DataCopyPadExtParams<SampleBFloat16>` 等 SDK 类型实例化失败。

正确做法：host 侧使用 `SampleBFloat16` 做计算，launch 时映射为 `bfloat16_t`；或直接用非模板入口函数硬编码 kernel 类型。

### 7. 抽 tiling 时把”对齐分支”和”非对齐分支”机械合并
看起来像是在“清理重复代码”，但这类改法最容易偷偷改掉原始切块语义。

典型危险动作：
- 把只有非 32B 对齐路径才该用的 `AlignUp(baseColLen, ubMinBlockLen)`
- 也套到 32B 对齐路径上

结果往往不是立即编译错，而是：
- tiling 结果漂移
- blockDim / base shape 改掉
- correctness 或性能悄悄偏离上游实现

---

## 参考案例

以下两个案例展示了两种典型的迁移模式。处理其他算子时，按同样的方法论识别本算子的依赖结构，套用对应的模式即可——不要局限于案例中的具体符号名和文件名。

### 案例 1：add_rms_norm arch35（重外部依赖型——核心工作在依赖闭包提取）

当源目录是：
- `add_rms_norm/arch35/add_rms_norm_regbase.h`
- `add_rms_norm/arch35/add_rms_norm_regbase_common.h`
- `add_rms_norm/arch35/add_rms_norm_regbase_split_d.h`

典型处理方式是：
- 保留这 3 个 kernel 头
- 新增 `add_rms_norm_local_deps.h`
- 删除相对 include：
  - `../inc/platform.h`
  - `../../rms_norm/rms_norm_base.h`
  - `../../rms_norm/arch35/rms_norm_regbase_common.h`
  - `../../norm_common/reduce_common_regbase.h`
- 本地化最小依赖：
  - `GetUbBlockSize` / `GetVRegSize`
  - `BUFFER_NUM` / `DOUBLE_BUFFER_NUM` / `ONCE_VECTOR_SIZE` / `CeilDiv` / `is_same`
  - `V_LENGTH` / `CeilAlign` / `ComputeMultiLevelRstd` / `castTraitB162B32` / `castTraitB322B16`
  - `DataCopyImpl` / `ComputeSum` / `ComputeRstd` / `ComputeMultiLevelReduce`
- 修正命名空间漂移：
  - `castTraitB162B32` / `castTraitB322B16` 应归到 `NormCommon`
- 如同时落地 `add_rms_norm_apt.cpp`，要额外声明：
  - kernel 依赖已本地化
  - 但 `DTYPE_X1` / `GET_TILING_DATA_WITH_STRUCT` 仍来自外部编译环境
- 如果继续把入口做成可直接被其他 kernel/调用方 include 的版本，优先按下面方式改：
  - 把 `*_apt.cpp` 改成 `*_apt.h`
  - 按 tiling key 拆成两个独立入口函数，而不是单一 `add_rms_norm(...)` 分发
  - 不再使用 `GET_TILING_DATA_WITH_STRUCT` / `TILING_KEY_IS`
  - 两种 tiling struct 直接作为函数参数传入
  - 如果用户只要求“增加模板参数”，默认只给入口函数增加 `DTYPE_X1 / DTYPE_X2 / DTYPE_GAMMA / DTYPE_Y / DTYPE_RSTD / DTYPE_X` 这类模板参数，**参数声明仍保持原来的 `GM_ADDR`**，不要擅自改成 `__gm__ T*` 形式
  - 入口头 `*_apt.h` 默认不要写 `using namespace`；优先在入口实现里使用 `::AscendC::...`、`::AddRmsNorm::...` 这类显式限定，避免把命名空间污染扩散给包含方
  - 头文件化后的 kernel 实现默认再用 `#if defined(__NPU_DEVICE__) ... #endif` 包起来，避免 host 侧包含入口头时直接编译 device 实现
  - 但如果用户明确指出宏保护应该放在 `add_rms_norm_local_deps.h / add_rms_norm_regbase_common.h / add_rms_norm_regbase.h / add_rms_norm_regbase_split_d.h` 这类 **kernel 实现头**，就把 `__NPU_DEVICE__` 宏放在这些实现头里，而不是优先放在入口头 `*_apt.h`

---

### 案例 2：swi_glu arch35（轻外部依赖型——核心工作在入口改造 + tiling 提取）

这是一个 **kernel 已自包含（无 `..` include）+ 完整 tiling + host glue** 的案例。与 add_rms_norm 不同，swi_glu arch35 的 kernel 头本身没有外部相对 include，核心工作在入口改造和 tiling 提取。

### 源文件集合

kernel 源（arch35 目录）：
- `glu_tiling.hpp` — `AlignUp`/`ISMAX` 工具函数 + 未使用的 struct
- `glu_tiling_kernel.hpp` — `SwigluSingleTilingKernel` + 过程宏
- `swi_glu_impl.hpp` — `SwigluVector`（fp32 计算路径）
- `swi_glu_bf16.hpp` — `SwigluVectorBF16`（fp16/bf16 路径，float 中间精度）
- `swi_glu_single.hpp` — `SwigluSingle`（tiling 编排 + 数据搬运）

入口源（父目录）：
- `swi_glu_apt.cpp` — 单一 `extern "C"` 入口 + `GET_TILING_DATA` + `TILING_KEY_IS`

tiling 源（op_host 目录）：
- `swi_glu_tiling.h` — `BEGIN_TILING_DATA_DEF(SwiGluTilingData)` + 14 个字段
- `swi_glu_tiling.cpp` — `GluSingleTilingCalculator` 类 + `IMPL_OP_OPTILING` 注册

### 适配后文件集合

```
swi_glu_story/
  include/
    kernel/
      swi_glu_local_deps.h      ← 新增：kernel_operator.h + simt_api/asc_bf16.h
      glu_tiling.hpp             ← 删除 GluTilingData/GluSingleTilingData/MAX_CORE_NUMBER
      glu_tiling_kernel.hpp      ← include 改为 local_deps.h + swi_glu_tiling.h；stride u16→u32
      swi_glu_impl.hpp           ← include 改为 local_deps.h
      swi_glu_bf16.hpp           ← include 改为 local_deps.h
      swi_glu_single.hpp         ← 加 (void)beta_gm; 消除未使用警告
      swi_glu_apt.h              ← 新增：3 个独立入口 + KERNEL_TASK_TYPE_DEFAULT
    tiling/
      swi_glu_tiling.h           ← 新增：plain C++ struct，仅 6 字段
      swi_glu_host_tiling.h      ← 新增：从 tiling.cpp 提取计算逻辑
  src/
    swi_glu_story.cpp             ← 新增：standalone host 驱动
  CMakeLists.txt
```

### 关键改造点

**1. TilingData 替换**
```cpp
// 原始（14 字段 + getter/setter）
BEGIN_TILING_DATA_DEF(SwiGluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
    // ... 14 fields total
END_TILING_DATA_DEF;

// 适配后（仅 kernel 使用的 6 字段，直接转成普通 C++ struct）
struct SwiGluTilingData {
    uint32_t is32BAligned = 0;
    uint32_t isDoubleBuffer = 0;
    uint64_t rowLen = 0;
    uint64_t colLen = 0;
    uint32_t baseRowLen = 0;
    uint32_t baseColLen = 0;
};
```

**2. 入口改造（apt.cpp → apt.h）**
```cpp
// 原始：单入口 + 运行时类型分发
extern "C" __global__ __aicore__ void swi_glu(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(1)) { /* fp16 */ }
    else if (TILING_KEY_IS(0)) { /* fp32 */ }
}

// 适配后：按 tilingKey（此处是 dtype）拆 3 个入口，tiling by value
__global__ __aicore__ void swi_glu_fp16(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const SwiGluTilingData tilingData) {
    (void)workspace;
    if (tilingData.isDoubleBuffer == 1) { SwiGluRunBf16Impl<half, float, half, 2>(...); }
    else { SwiGluRunBf16Impl<half, float, half, 1>(...); }
}
```

**3. Host tiling 提取**
- 去掉 `gert::TilingContext*`、`OP_LOGE`、`IMPL_OP_OPTILING`
- 用 `platform_ascendc::PlatformAscendCManager::GetInstance()` 获取 UB/core 信息
- 返回 `SwiGluLaunchConfig{blockDim, workspaceSize, dtype, tiling}`
- tiling 计算逻辑（`CalcOptBaseShape` 等）逐行保真搬运，保持对齐/非对齐分支独立
- 只保留 `SWIGLU_SINGLE` 路径，去掉 `SWIGLU_GRAD_SINGLE`

**4. Host 驱动**
- 内置确定性数据生成（线性递推 + 取模，不依赖 Python/numpy）
- C++ golden：`silu(x) * y = x / (1 + exp(-x)) * y`
- 测试 3 种 dtype（fp16, fp32, bf16），各有独立容差
- `GM_ADDR` 贯穿 host/launch/kernel 三侧

### 如何判断你的算子属于哪种模式

| 判断维度 | 重外部依赖型（类似案例 1） | 轻外部依赖型（类似案例 2） |
|------|-------------|---------|
| kernel `..` include 数量 | 多（3+ 个） | 少或无 |
| 核心工作量 | 依赖闭包提取 | 入口改造 + tiling 提取 |
| local_deps.h 内容 | 装满搬运过来的外部函数/常量 | 仅集中 SDK include |
| 入口拆分依据 | 按算子自身 tilingKey 的划分维度（计算模式、dtype 等） | 同左 |
| tiling 提取 | 视需求 | 通常需要完整提取 |

多数算子会混合两种模式的特征——先统计 `..` include 数量和 tilingKey 分发逻辑，就能判断主要工作量在哪。

---

## Host 驱动代码模板

执行适配时，**必须基于以下模板生成 host 驱动代码，不要去读目标仓库（如 cann-samples）里的其他 story cpp 文件**。模板是算子无关的，适配时只需替换算子特定部分（标注为 `/* OPERATOR-SPECIFIC */` 的位置）。

### 完整 host 驱动模板

```cpp
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kernel_operator.h"
/* OPERATOR-SPECIFIC: include kernel 入口头和 host tiling 头 */
// #include "xxx_apt.h"
// #include "xxx_host_tiling.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

/* OPERATOR-SPECIFIC: 问题规模常量 */
// constexpr size_t kRowLen = 128;
// constexpr size_t kColLen = 256;

constexpr int kMaxErrorElemNum = 10;
constexpr float kFloatTolerance = 1e-4f;
constexpr float kHalfTolerance = 5e-2f;
constexpr float kBFloat16Tolerance = 5e-2f;

// ---- BFloat16 host 侧辅助（AscendC bfloat16_t 仅 device 可用）----
using SampleHalf = half;
struct SampleBFloat16 {
    uint16_t bits;
};

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    const uint32_t roundingBias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float Bf16BitsToFloat(uint16_t bits)
{
    const uint32_t value = static_cast<uint32_t>(bits) << 16;
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// ---- 类型转换 trait ----
template <typename T>
T FromFloat(float value) { return static_cast<T>(value); }

template <>
SampleBFloat16 FromFloat<SampleBFloat16>(float value) { return SampleBFloat16{FloatToBf16Bits(value)}; }

template <typename T>
float ToFloat(T value) { return static_cast<float>(value); }

template <>
float ToFloat<SampleBFloat16>(SampleBFloat16 value) { return Bf16BitsToFloat(value.bits); }

template <typename T> float GetTolerance();
template <> float GetTolerance<float>()           { return kFloatTolerance; }
template <> float GetTolerance<SampleHalf>()      { return kHalfTolerance; }
template <> float GetTolerance<SampleBFloat16>()  { return kBFloat16Tolerance; }

template <typename T> const char *GetDtypeName();
template <> const char *GetDtypeName<float>()           { return "float32"; }
template <> const char *GetDtypeName<SampleHalf>()      { return "float16"; }
template <> const char *GetDtypeName<SampleBFloat16>()  { return "bfloat16"; }

// ---- ACL 基础 ----
void CheckAcl(aclError ret, const char *msg)
{
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << msg << " failed. aclError=" << ret;
        throw std::runtime_error(oss.str());
    }
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ---- 结果比对 ----
template <typename T>
size_t CompareBuffer(const std::string &name, const std::vector<T> &actual,
    const std::vector<T> &expected, float atol)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " size mismatch");
    }
    size_t mismatchCount = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float act = ToFloat(actual[i]);
        const float exp = ToFloat(expected[i]);
        const float err = std::fabs(act - exp);
        maxAbsErr = std::max(maxAbsErr, err);
        if (err > atol) {
            if (mismatchCount < static_cast<size_t>(kMaxErrorElemNum)) {
                std::cout << name << " mismatch[" << i << "]: expected=" << exp
                          << ", actual=" << act << ", abs_err=" << err << std::endl;
            }
            ++mismatchCount;
        }
    }
    std::cout << name << ": total=" << actual.size() << ", mismatch=" << mismatchCount
              << ", max_abs_err=" << maxAbsErr << std::endl;
    return mismatchCount;
}

/* OPERATOR-SPECIFIC: 数据生成 —— 确定性，不依赖 Python/numpy */
// template <typename T>
// void BuildInput(std::vector<T> &data, size_t rowLen, size_t colLen) { ... }

/* OPERATOR-SPECIFIC: Golden 计算 —— 用 float 精度算完再转回目标类型 */
// template <typename T>
// void ComputeReference(const std::vector<T> &input, std::vector<T> &output) { ... }

/* OPERATOR-SPECIFIC: Kernel launch 分发 —— 按 tilingKey 选择入口 */
// template <typename T>
// void LaunchKernel(GM_ADDR inputDevice, GM_ADDR outputDevice, GM_ADDR workspaceDevice,
//     const XxxLaunchConfig &launchConfig, aclrtStream stream)
// {
//     if constexpr (std::is_same_v<T, float>) {
//         xxx_fp32<<<launchConfig.blockDim, 0, stream>>>(inputDevice, outputDevice, workspaceDevice, launchConfig.tiling);
//     } else if constexpr (std::is_same_v<T, SampleHalf>) {
//         xxx_fp16<<<launchConfig.blockDim, 0, stream>>>(inputDevice, outputDevice, workspaceDevice, launchConfig.tiling);
//     } else { ... }
// }

/* OPERATOR-SPECIFIC: 单次测试流程 */
template <typename T>
void RunOneCase(aclrtStream stream)
{
    /* 1. 计算 size */
    // const size_t inputSize = ...;
    // const size_t outputSize = ...;

    /* 2. 生成输入 + golden */
    // std::vector<T> inputData;
    // BuildInput(inputData, ...);
    // std::vector<T> outputGolden;
    // ComputeReference(inputData, outputGolden);
    // std::vector<T> outputActual(outputElemNum, FromFloat<T>(0.0f));

    /* 3. 计算 tiling + launch config */
    // auto launchConfig = XxxHost::CalcLaunchConfig(...);

    /* 4. 设备内存分配 —— GM_ADDR 贯穿 */
    GM_ADDR inputDevice = nullptr;
    GM_ADDR outputDevice = nullptr;
    GM_ADDR workspaceDevice = nullptr;

    // CheckAcl(aclrtMalloc((void **)&inputDevice, inputSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc input");
    // CheckAcl(aclrtMalloc((void **)&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc output");
    // CheckAcl(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");

    try {
        /* 5. H2D 拷贝 */
        // CheckAcl(aclrtMemcpy(inputDevice, inputSize, inputData.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE), "H2D input");

        /* 6. Launch + Sync */
        // LaunchKernel<T>(inputDevice, outputDevice, workspaceDevice, launchConfig, stream);
        // CheckAcl(aclrtSynchronizeStream(stream), "sync");

        /* 7. D2H 拷贝 + 验证 */
        // CheckAcl(aclrtMemcpy(outputActual.data(), outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST), "D2H output");
        // size_t mismatch = CompareBuffer(GetDtypeName<T>(), outputActual, outputGolden, GetTolerance<T>());
        // if (mismatch != 0) { throw std::runtime_error("result check failed"); }
        // std::cout << GetDtypeName<T>() << " run succeeded" << std::endl;
    } catch (...) {
        if (workspaceDevice != nullptr) { aclrtFree(workspaceDevice); }
        if (outputDevice != nullptr) { aclrtFree(outputDevice); }
        if (inputDevice != nullptr) { aclrtFree(inputDevice); }
        throw;
    }

    CheckAcl(aclrtFree(workspaceDevice), "free workspace");
    CheckAcl(aclrtFree(outputDevice), "free output");
    CheckAcl(aclrtFree(inputDevice), "free input");
}

void RunSample(aclrtStream stream)
{
    /* OPERATOR-SPECIFIC: 按需测试各种 dtype */
    RunOneCase<SampleHalf>(stream);
    RunOneCase<float>(stream);
    RunOneCase<SampleBFloat16>(stream);
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    try {
        RunSample(stream);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        ret = 1;
    }

    Finalize(deviceId, stream);
    return ret;
}
```

### 模板使用说明

适配时只需要替换 `/* OPERATOR-SPECIFIC */` 标注的部分：

| 替换点 | 说明 |
|--------|------|
| include 头 | 替换为实际的 `xxx_apt.h` 和 `xxx_host_tiling.h` |
| 问题规模常量 | 根据算子设定合理的测试 shape |
| `BuildInput` | 确定性数据生成（线性递推 + 取模），不依赖 Python |
| `ComputeReference` | 用 float 精度计算 golden，再转回目标类型 |
| `LaunchKernel` | 按 tilingKey 分发到各入口函数 |
| `RunOneCase` | 填入实际的 size 计算、tiling 调用、内存拷贝 |

以下部分 **不需要修改**，直接复用：
- ACL init/finalize
- BFloat16 host 辅助（`FloatToBf16Bits`/`Bf16BitsToFloat`/`SampleBFloat16`）
- 类型转换 trait（`FromFloat`/`ToFloat`/`GetTolerance`/`GetDtypeName`）
- `CheckAcl` 错误检查
- `CompareBuffer` 结果比对
- `main` 函数骨架

### CMakeLists.txt 模板

CMakeLists.txt 可以参考目标仓库的 cmake 体系

---

## 工具偏好

优先使用：
- `Glob`：找文件
- `Grep`：找 include 和符号引用
- `Read`：读源文件和定义文件
- `Edit`：改现有文件
- `Write`：只用于新增 helper 头或新增目标副本

不要用 shell 的 `grep/cat/find` 代替这些专用工具，除非确实没有替代方案。

---

## 一句话原则

这类任务的核心不是”把文件复制过来”，而是：

**把 kernel 代码适配成一个最小依赖闭包明确、语义不变、目标目录内可维护的本地版本。**

**执行本 skill 时，只读原始算子源码（kernel + tiling），不读目标仓库中的其他 sample / story / utils。所有 host 样板代码从本 skill 内嵌模板生成。**
