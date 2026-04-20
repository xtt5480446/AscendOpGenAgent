# precision-tuning Agent 迁移实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 OpenOps-debug 的 precision-tuning skill fork 到 AscendOpGenAgent，适配新仓库的路径约定、评测流程和 kernel 结构，使其可作为独立的 post-generation precision debug agent 运行。

**Architecture:** Fork + 深度适配。precision_forensics.py 改为调用 verification_ascendc.py 获取数值差异数据；precision_gate.py 改写 kernel 检查逻辑；SKILL.md 替换 Phase A 参考文件、删除 DSL 分析、统一评测入口为 evaluate_ascendc.sh。

**Tech Stack:** Python 3.8+，opencode agent framework，AscendC / CANN

**Source files (OpenOps-debug):**
- `.opencode/skills/precision-tuning/scripts/precision_forensics.py` (892 lines)
- `.opencode/skills/precision-tuning/scripts/precision_gate.py` (1042 lines)
- `.opencode/skills/precision-tuning/scripts/precision_knowledge.py` (549 lines)
- `.opencode/skills/precision-tuning/SKILL.md` (825 lines)
- `.opencode/skills/precision-tuning/references/precision_knowledge_base.json`
- `.opencode/agents/precision-tuning.md`

**Target repo:** `/Users/junming/code/operator/AscendOpGenAgent/`

---

## Task 1: 创建目录结构，复制原始文件

**Files:**
- Create: `skills/ascendc/precision-tuning/references/`
- Create: `skills/ascendc/precision-tuning/scripts/`
- Copy: 上述所有源文件到目标位置

**Step 1: 创建目录**
```bash
mkdir -p /Users/junming/code/operator/AscendOpGenAgent/skills/ascendc/precision-tuning/references
mkdir -p /Users/junming/code/operator/AscendOpGenAgent/skills/ascendc/precision-tuning/scripts
```

**Step 2: 复制文件（作为初始基础，后续各 Task 修改）**
```bash
SRC=/Users/junming/code/operator/OpenOps/OpenOps-debug/.opencode/skills/precision-tuning
DST=/Users/junming/code/operator/AscendOpGenAgent/skills/ascendc/precision-tuning

cp $SRC/scripts/precision_forensics.py $DST/scripts/
cp $SRC/scripts/precision_gate.py $DST/scripts/
cp $SRC/scripts/precision_knowledge.py $DST/scripts/
cp $SRC/SKILL.md $DST/
cp $SRC/references/precision_knowledge_base.json $DST/references/
```

**Step 3: 验证文件存在**
```bash
ls /Users/junming/code/operator/AscendOpGenAgent/skills/ascendc/precision-tuning/scripts/
ls /Users/junming/code/operator/AscendOpGenAgent/skills/ascendc/precision-tuning/references/
```
Expected: 3 个 .py 文件 + precision_knowledge_base.json

**Step 4: Commit**
```bash
cd /Users/junming/code/operator/AscendOpGenAgent
git add skills/ascendc/precision-tuning/
git commit -m "feat: scaffold precision-tuning skill directory (pre-adaptation)"
```

---

## Task 2: 适配 precision_forensics.py

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py`

核心改动：
1. 删除 `setup_ascend_environment()` 及其所有调用（AscendOpGenAgent 的验证由 evaluate_ascendc.sh 负责环境设置）
2. 替换 `OperatorExecutor.load_and_execute()` — 改为调用 `verification_ascendc.py` 并解析 stdout
3. 简化 `OperatorTypeDetector.detect()` — 移除 `op_desc.json` 路径，只保留名称启发式
4. 更新路径常量和 `available_files` 字段
5. 更新 argparse（`--output-path` 改为 `--task-name`，repo root 自动从脚本位置推断）

**Step 1: 替换路径常量和 argparse（文件顶部）**

找到并替换：
```python
# 原
def setup_ascend_environment(op_name: str, output_path: str) -> str | None:
    ...  # 整个函数（约 100 行）删除
```

在文件顶部 import 区域后，增加 repo root 推断：
```python
# 新增：从脚本位置推断 repo root
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
SKILL_DIR  = SCRIPT_DIR.parent                        # precision-tuning/
REPO_ROOT  = SKILL_DIR.parent.parent.parent           # AscendOpGenAgent/
VERIF_SCRIPT = REPO_ROOT / "utils" / "verification_ascendc.py"
```

**Step 2: 替换 `OperatorExecutor` 类**

找到 `class OperatorExecutor:` 整个类（约 60 行），替换为：

```python
class OperatorExecutor:
    """调用 verification_ascendc.py 并解析 stdout 获取数值差异数据。"""

    def __init__(self, op_name: str, task_dir: str):
        self.op_name = op_name
        self.task_dir = Path(task_dir).resolve()

    def load_and_execute(self) -> dict:
        """
        运行 verification_ascendc.py，解析结构化 stdout。
        返回 {"ref_outputs": [...], "new_outputs": [...], "input_tensors": [...],
               "comparisons": [...], "all_passed": bool}
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, str(VERIF_SCRIPT), self.op_name],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")},
        )
        stdout = result.stdout
        comparisons = self._parse_comparisons(stdout)
        inputs_meta = self._parse_inputs(stdout)
        return {
            "stdout": stdout,
            "comparisons": comparisons,      # list of {case_idx, output_idx, ok, mismatch_ratio, max_abs_diff, mean_abs_diff, dtype, first_mismatch}
            "inputs_meta": inputs_meta,      # list of {name, shape, dtype}
            "all_passed": result.returncode == 0,
        }

    def _parse_comparisons(self, stdout: str) -> list:
        """
        解析形如:
          case[0]: output[0]: dtype(ref=float16, cand=float16), unequal_elements=37,
                   mismatch_ratio=0.289062%, max_abs_diff=0.00390625, mean_abs_diff=0.00390625
        的行。
        """
        import re
        results = []
        pattern = re.compile(
            r"case\[(\d+)\]: output\[(\d+)\]: "
            r"(?:dtype\(ref=([^,]+), cand=[^)]+\), )?"
            r"(?:unequal_elements=(\d+), )?"
            r"mismatch_ratio=([0-9.]+)%, "
            r"max_abs_diff=([0-9.eE+\-g]+), "
            r"mean_abs_diff=([0-9.eE+\-g]+)"
        )
        matched_pattern = re.compile(r"case\[(\d+)\]: output\[(\d+)\]: matched")
        for line in stdout.splitlines():
            m = pattern.search(line)
            if m:
                results.append({
                    "case_idx": int(m.group(1)),
                    "output_idx": int(m.group(2)),
                    "dtype": m.group(3) or "unknown",
                    "ok": False,
                    "mismatch_ratio": float(m.group(5)) / 100.0,
                    "max_abs_diff": float(m.group(6)),
                    "mean_abs_diff": float(m.group(7)),
                })
                continue
            m2 = matched_pattern.search(line)
            if m2:
                results.append({
                    "case_idx": int(m2.group(1)),
                    "output_idx": int(m2.group(2)),
                    "ok": True,
                    "mismatch_ratio": 0.0,
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                })
        return results

    def _parse_inputs(self, stdout: str) -> list:
        """解析 Inputs 段，提取 shape/dtype 用于 L6 布局分析。"""
        import re
        results = []
        in_inputs = False
        for line in stdout.splitlines():
            if line.strip() == "Inputs":
                in_inputs = True
                continue
            if line.startswith("-" * 10) and in_inputs:
                in_inputs = False
                continue
            if in_inputs:
                m = re.search(r"(inputs\[\d+\].*?): Tensor\(shape=(\([^)]*\)), dtype=(\w+)", line)
                if m:
                    results.append({"name": m.group(1), "shape": m.group(2), "dtype": m.group(3)})
        return results
```

**Step 3: 更新 `OperatorTypeDetector.detect()`**

找到该方法中读取 `op_desc.json` 的部分（约第 171-188 行），删除整个 `# 1. 尝试从 op_desc.json 读取` 块，只保留名称启发式推断：

```python
def detect(self, op_name: str, task_dir: str) -> dict:
    """基于算子名做关键词匹配推断算子类型。置信度低时标注 name_heuristic。"""
    name = op_name.lower()
    type_map = [
        (["pool"], "pooling"),
        (["norm", "rms", "layer"], "normalization"),
        (["matmul", "gemm", "linear", "quant_matmul"], "matmul"),
        (["gather", "scatter", "index"], "gather"),
        (["concat", "cat"], "concat"),
        (["attn", "attention", "softmax"], "attention"),
        (["relu", "gelu", "silu", "activation", "elementwise"], "elementwise"),
    ]
    for keywords, op_type in type_map:
        if any(kw in name for kw in keywords):
            return {"op_type": op_type, "source": "name_heuristic", "confidence": "low"}
    return {"op_type": "unknown", "source": "name_heuristic", "confidence": "low"}
```

**Step 4: 更新 `PrecisionForensics.__init__` 和主入口**

找到 `class PrecisionForensics:` 的 `__init__`，更新：
```python
def __init__(self, op_name: str, task_dir: str, attempt: int = 0):
    self.op_name = op_name
    self.task_dir = task_dir      # 替换 output_path
    self.attempt = attempt
    self.tuning_dir = os.path.join(task_dir, "precision_tuning")
    os.makedirs(self.tuning_dir, exist_ok=True)
```

找到 `available_files` 部分（约第 839-844 行），替换：
```python
"available_files": {
    "reference": os.path.exists(os.path.join(self.task_dir, "model.py")),
    "custom": os.path.exists(os.path.join(self.task_dir, "model_new_ascendc.py")),
    # dsl / op_desc 字段已移除
},
```

更新 `OperatorExecutor` 调用处（约第 798 行）：
```python
# 原
data = OperatorExecutor(self.op_name, self.output_path).load_and_execute()
# 新
data = OperatorExecutor(self.op_name, self.task_dir).load_and_execute()
```

更新 `OperatorTypeDetector` 调用处（约第 797 行）：
```python
# 原
op_type_info = OperatorTypeDetector().detect(self.op_name, self.output_path)
# 新
op_type_info = OperatorTypeDetector().detect(self.op_name, self.task_dir)
```

更新 argparse（文件末尾）：
```python
# 原
parser.add_argument("op_name", ...)
parser.add_argument("--output-path", required=True, ...)
# 新
parser.add_argument("task_name", help="task 目录名（相对于 repo root）")
parser.add_argument("--task-dir", default=None,
                    help="task 绝对路径，默认为 {REPO_ROOT}/{task_name}")
parser.add_argument("--attempt", type=int, default=0)

# main() 内
task_dir = args.task_dir or str(REPO_ROOT / args.task_name)
PrecisionForensics(args.task_name, task_dir, args.attempt).run()
```

**Step 5: 验证脚本语法**
```bash
cd /Users/junming/code/operator/AscendOpGenAgent
python -c "import ast; ast.parse(open('skills/ascendc/precision-tuning/scripts/precision_forensics.py').read()); print('OK')"
```
Expected: `OK`

**Step 6: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "feat: adapt precision_forensics to AscendOpGenAgent path conventions"
```

---

## Task 3: 重写 precision_gate.py

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_gate.py`

核心改动：
1. 删除 `_find_project_dir()` — 替换为 `_kernel_dir()`
2. 重写 `check_fix()` Gate-X — 检查 pybind11.cpp、PYBIND11_MODULE、非 pybind .cpp
3. 重写 `_check_prerequisite_code()` — 对齐新路径
4. 新增 `_check_import_name_match()` — 比对 model_new_ascendc.py 的 import 名与 PYBIND11_MODULE
5. 更新所有 `output_path` → `task_dir` 参数名

**Step 1: 添加 REPO_ROOT 推断（文件顶部）**

在 import 区域后增加：
```python
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent.parent   # AscendOpGenAgent/
```

**Step 2: 替换 `_find_project_dir()` 为 `_kernel_dir()`**

找到 `def _find_project_dir(self)` （约第 979 行），整个替换：
```python
def _kernel_dir(self) -> str | None:
    """返回 {task_dir}/kernel/ 路径，不存在则返回 None。"""
    kdir = os.path.join(self.task_dir, "kernel")
    return kdir if os.path.isdir(kdir) else None

def _check_import_name_match(self) -> bool:
    """比对 model_new_ascendc.py 的 import _xxx_ext 名与 pybind11.cpp 的 PYBIND11_MODULE 名。"""
    import re
    wrapper = os.path.join(self.task_dir, "model_new_ascendc.py")
    pybind  = os.path.join(self.task_dir, "kernel", "pybind11.cpp")
    if not os.path.exists(wrapper) or not os.path.exists(pybind):
        return False
    # 从 wrapper 提取 import _xxx_ext
    wrapper_text = open(wrapper).read()
    import_m = re.search(r"import\s+(_\w+)", wrapper_text)
    if not import_m:
        return False
    import_name = import_m.group(1)
    # 从 pybind11.cpp 提取 PYBIND11_MODULE(xxx, ...)
    pybind_text = open(pybind).read()
    module_m = re.search(r"PYBIND11_MODULE\s*\(\s*(\w+)\s*,", pybind_text)
    if not module_m:
        return False
    module_name = "_" + module_m.group(1)   # pybind 宏名前加下划线
    return import_name == module_name
```

**Step 3: 重写 `check_fix()` Gate-X**

找到 `def check_fix(self)` 整个方法（约第 185-216 行），替换：
```python
def check_fix(self) -> dict:
    """Gate-X: 验证代码文件完整性（修复后调用）。"""
    prereq = self._check_prerequisite_audit()
    if not prereq["satisfied"]:
        checks = {"prerequisite_audit": False}
        checks.update(prereq["detail"])
        return self._result("GATE-X", checks)

    kdir = self._kernel_dir()
    pybind_path = os.path.join(kdir, "pybind11.cpp") if kdir else None
    has_pybind = kdir is not None and os.path.exists(pybind_path)
    has_module_macro = False
    has_non_pybind_cpp = False
    import_name_ok = False

    if has_pybind:
        content = open(pybind_path).read()
        has_module_macro = "PYBIND11_MODULE" in content
        has_non_pybind_cpp = any(
            f.endswith(".cpp") and f != "pybind11.cpp"
            for f in os.listdir(kdir)
        )
        import_name_ok = self._check_import_name_match()

    checks = {
        "prerequisite_audit": True,
        "kernel_dir_exists": kdir is not None,
        "pybind11_cpp_exists": has_pybind,
        "pybind11_has_module_macro": has_module_macro,
        "has_non_pybind_cpp": has_non_pybind_cpp,
        "model_new_ascendc_exists": os.path.exists(
            os.path.join(self.task_dir, "model_new_ascendc.py")
        ),
        "import_name_consistent": import_name_ok,
    }
    return self._result("GATE-X", checks)
```

**Step 4: 重写 `_check_prerequisite_code()`**

找到该方法（约第 313-325 行），替换：
```python
def _check_prerequisite_code(self) -> dict:
    """检查 kernel/pybind11.cpp 存在且非空。"""
    kdir = self._kernel_dir()
    if not kdir:
        return {"satisfied": False,
                "reason": f"{self.task_dir}/kernel/ 不存在",
                "detail": {"kernel_dir_exists": False}}
    pybind = os.path.join(kdir, "pybind11.cpp")
    if not os.path.exists(pybind) or os.path.getsize(pybind) < 100:
        return {"satisfied": False,
                "reason": f"{pybind} 不存在或内容过少",
                "detail": {"pybind11_cpp_exists": False}}
    return {"satisfied": True, "reason": "", "detail": {}}
```

**Step 5: 全文替换参数名**

将所有 `self.output_path` 替换为 `self.task_dir`，`output_path` 参数名替换为 `task_dir`（包括 `__init__`、argparse、`tuning_dir` 路径构造）：

```python
# __init__ 中
self.task_dir = task_dir           # 原 self.output_path = output_path
self.tuning_dir = os.path.join(task_dir, "precision_tuning")
```

argparse 末尾：
```python
# 原
parser.add_argument("--output-path", required=True)
# 新
parser.add_argument("--task-name", required=True, help="task 目录名")
parser.add_argument("--task-dir", default=None, help="task 绝对路径，默认 {REPO_ROOT}/{task_name}")

# main() 内
task_dir = args.task_dir or str(REPO_ROOT / args.task_name)
PrecisionGate(args.task_name, task_dir, args.attempt).run()
```

**Step 6: 验证语法**
```bash
python -c "import ast; ast.parse(open('skills/ascendc/precision-tuning/scripts/precision_gate.py').read()); print('OK')"
```

**Step 7: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/precision_gate.py
git commit -m "feat: rewrite precision_gate kernel checks for AscendOpGenAgent structure"
```

---

## Task 4: 适配 precision_knowledge.py

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_knowledge.py`

改动极小：只更新 `--output-path` → `--task-dir` / `--task-name` 参数名，以及路径推断（与 Task 2/3 保持一致）。

**Step 1: 更新 argparse 和 REPO_ROOT 推断**

在文件顶部增加：
```python
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
```

找到 argparse 中的 `--output-path`，替换：
```python
# 原
parser.add_argument("--output-path", required=True)
# 新
parser.add_argument("--task-name", required=True)
parser.add_argument("--task-dir", default=None)
```

在 main() 内补充：
```python
task_dir = args.task_dir or str(REPO_ROOT / args.task_name)
# 后续所有 args.output_path → task_dir
```

**Step 2: 验证语法**
```bash
python -c "import ast; ast.parse(open('skills/ascendc/precision-tuning/scripts/precision_knowledge.py').read()); print('OK')"
```

**Step 3: Commit**
```bash
git add skills/ascendc/precision-tuning/scripts/precision_knowledge.py
git commit -m "feat: update precision_knowledge argparse to task-name convention"
```

---

## Task 5: 重写 SKILL.md

**Files:**
- Modify: `skills/ascendc/precision-tuning/SKILL.md`

这是改动最多的文件，按节分步处理。

### 5.1 更新 frontmatter 和参数说明

找到文件头部的 `Prerequisites` / 参数说明块，替换为：

```markdown
## Prerequisites

- `{task_dir}/model.py` — 参考实现（含 Model, get_input_groups/get_inputs, get_init_inputs）
- `{task_dir}/model_new_ascendc.py` — AscendC wrapper（含 ModelNew）
- `{task_dir}/kernel/pybind11.cpp` — host launch + pybind
- `{task_dir}/kernel/{op_name}_tiling.h` — TilingData 定义
- `{task_dir}/kernel/*.cpp` + `*_kernel.h` — 至少一个非 pybind kernel 文件
- `{task_dir}/{op_name}.json` — 测试用例（JSON Lines）

其中 `task_dir = {repo_root}/{task_name}`，`repo_root` 为 AscendOpGenAgent 仓库根目录。
```

### 5.2 替换 Step 0 快照路径

找到 Step 0 中所有 `{OpName}Custom/op_kernel/{op_name}_custom.cpp` 等路径，替换为：

```bash
# 保存快照时，复制以下文件：
cp "{task_dir}/kernel/pybind11.cpp" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/pybind11.cpp"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/model_new_ascendc.py"
# kernel/*.cpp 和 *_kernel.h 全部复制
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/kernel/"
```

所有 attempt_N 快照、baseline 快照、current_best 快照同样替换。

### 5.3 替换 Step 1 取证命令

找到：
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_forensics.py \
    {op_name} --output-path "{output_path}" --attempt {attempt}
```
替换为：
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_forensics.py \
    {task_name} --attempt {attempt}
```

Gate 验证命令同样替换路径前缀：
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step forensics --task-name {task_name} --attempt {attempt}
```

### 5.4 替换知识库检索命令

全文将：
```bash
python3 .opencode/skills/precision-tuning/scripts/precision_knowledge.py search \
    --kb-path .opencode/skills/precision-tuning/references/precision_knowledge_base.json \
```
替换为：
```bash
python3 skills/ascendc/precision-tuning/scripts/precision_knowledge.py search \
    --kb-path skills/ascendc/precision-tuning/references/precision_knowledge_base.json \
```

`--output-path` 参数替换为 `--task-name {task_name}`。

### 5.5 更新 Sub-step 2.2 — 删除 DSL 内容

找到 `#### Sub-step 2.2: 算子计算流程分解`，做以下修改：

**删除**：
- `{op_name}_dsl.py` 读取步骤（第 3 条读取项）
- `DSL tiling 策略摘要` 整个子块（n_cores、tile_length、归约维度完整性等字段）
- `DSL 文件状态` 字段
- 每个 Step 中的 `DSL 对应: dsl.py 中的 <xxx>` 行

**修改**：
```markdown
**读取** (按顺序):
1. `{task_dir}/model.py` — 参考实现的 forward() 逻辑
2. `{task_dir}/{op_name}_op_desc.json` — 如不存在则跳过
3. `.opencode/skills/precision-tuning/references/decomposition_examples/README.md` — 分解示例索引
```
改为：
```markdown
**读取** (按顺序):
1. `{task_dir}/model.py` — 参考实现的 forward() 逻辑
2. `archive_tasks/` 中最近似案例的 `model.py`（可选，用于对比计算链）
```

### 5.6 更新 Sub-step 2.3 Phase A — 替换参考文件

找到 Phase A 读取列表：
```
1. 根据 L8 op_type 路由，强制读取对应的 lowering 示例 ...
   - softmax → .opencode/skills/dsl-lowering/references/...
   - mse_loss → ...
2. 强制读取 error_correction_examples.md
3. 强制读取 non_aligned_example
4. 强制读取 tl_asc_routing.md
```

替换为：
```markdown
**Phase A 读取**:

1. 根据 L8 op_type 从 `archive_tasks/` 路由，读取对应案例的 `kernel/` 目录（仅含有完整 kernel/ 的案例）：
   - pooling → `archive_tasks/avg_pool3_d/kernel/`
   - normalization / rmsnorm / layernorm → `archive_tasks/rms_norm/kernel/`（含 vector_tile.h）
   - matmul / gemm / linear → `archive_tasks/matmul_leakyrelu/kernel/` 或 `archive_tasks/quant_matmul/kernel/`
   - gather / scatter / index → `archive_tasks/gather_elements_v2/kernel/`
   - concat / memory layout → `archive_tasks/concat_dv2/kernel/`
   - **attention / softmax** → 无有效 AscendC 案例（flash_attention 无 kernel/），跳过案例读取，仅读 API 文档
   - **纯 elementwise / activation** → 无纯向量案例，仅读 `dsl2Ascendc_compute_vector.md`
   - 无精确匹配 → 选最近似案例，在 [REFERENCE_IMPL_SPEC] 中标注"参考案例非精确匹配"

2. **必须读取**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc.md`
   （替代 error_correction_examples.md）

3. **必须读取**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`
   （替代 non_aligned_example，包含 DataCopyPad 触发条件）

4. **必须读取**: `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`
   （替代 tl_asc_routing.md）
   API 详细文档：`skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/`
```

### 5.7 更新 Sub-step 2.3 Phase B — 替换读取文件列表

找到 Phase B 读取：
```
1. {output_path}/{OpName}Custom/op_kernel/{op_name}_custom.cpp — Kernel 代码
2. {output_path}/{OpName}Custom/op_host/{op_name}_custom.cpp — Host 代码
3. {output_path}/{OpName}Custom/op_host/{op_name}_custom_tiling.h — TilingData
```

替换为：
```markdown
**Phase B 读取** (全部在 `{task_dir}/kernel/` 下):
1. `{op_name}_tiling.h` — TilingData 结构体定义
2. `*_kernel.h` — 所有 kernel 类定义（可能有多个）
3. `*.cpp`（排除 pybind11.cpp）— 所有 kernel entry 文件
4. `kernel_common.h`、`vector_tile.h`、`matmul_tile.h`（若存在）— helper 逻辑
5. `pybind11.cpp` — host tiling 计算、workspace 分配、launch 逻辑

注意：AscendOpGenAgent 中 host 逻辑（TilingFunc）在 pybind11.cpp 内，不是单独的 op_host.cpp。
```

同时，删除 `[KERNEL_STEP_TRACE]` 中的：
- `DSL 中对应参数` 行
- `Host vs DSL 是否一致` 行

### 5.8 更新 Step 4 — 替换编译/评测命令

找到 Step 4 的三步（install + compile + evaluate），合并替换为：

```markdown
### Step 4: 重新编译 + 精度验证

```bash
bash skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh {task_name}
```

**失败分类处理**（根据 stdout 判断）：

| 失败类型 | 特征 | 处理 |
|---|---|---|
| Infra 失败 | SSH 超时、docker exec 失败 | 停止，报告环境问题，不进入修复循环 |
| Build 失败 | `build_ascendc.py` 报编译错误 | 修复 kernel .cpp/.h，最多 3 次 |
| Import 失败 | import 阶段报 ModuleNotFoundError 或 PYBIND11_MODULE 名不一致 | 检查 model_new_ascendc.py import 名 vs pybind11.cpp |
| Numerical 失败 | verification_ascendc.py 报 mismatch | 进入 precision_forensics → 审计 → 修复循环 |
```

### 5.9 更新 Step 4.4 — 更新结果解析

`evaluate_ascendc.sh` 调用 `verification_ascendc.py`，其输出格式：
- PASS: `Result: pass`
- FAIL: `Result: fail` + 各 case 的 `mismatch_ratio=XX.XX%, max_abs_diff=X.XXX`

`validation_result_attempt_{attempt}.json` 写入逻辑不变，正则改为匹配新格式：
- `match_rate`: `r"mismatch_ratio=([0-9.]+)%"` 取所有 case 平均，或直接用最差 case
- `max_diff`: `r"max_abs_diff=([0-9.eE+\-g]+)"`

### 5.10 更新 Step 5 成功收尾 — 增加 trace-recorder 调用

在 Step 5.4 输出成功报告之前，增加：

```markdown
**5.3b 更新 trace.md（追加精度调优记录）**：

调用 `trace-recorder` skill 在 `{task_dir}/trace.md` 末尾追加 `## 精度调优` section：

```
## 精度调优

- 调优轮次: {attempt + 1}
- 最终 match_rate: {final_match_rate}%
- 最终 max_diff: {final_max_diff}
- 根因摘要: {root_cause_summary}
- 修复摘要: {fix_summary}
```
```

### 5.11 Step 6 失败报告同样追加 trace

在 Step 6 失败报告输出后，同样调用 trace-recorder 追加记录，字段补充：
- `loop_stop_reason`
- 每轮的 `fix_type + outcome`

**Step 12: 验证 SKILL.md 无明显遗漏（手动 grep）**
```bash
grep -n "dsl-lowering\|_reference\.py\|_custom\.py\|_op_desc\|OpName.*Custom\|op_kernel\|op_host\|generate_pybind\|evaluate\.py\b\|output_path" \
    skills/ascendc/precision-tuning/SKILL.md
```
Expected: 0 matches（所有旧路径已替换）

**Step 13: Commit**
```bash
git add skills/ascendc/precision-tuning/SKILL.md
git commit -m "feat: adapt SKILL.md for AscendOpGenAgent (paths, Phase A refs, eval pipeline)"
```

---

## Task 6: 创建 agents/precision-tuning.md

**Files:**
- Create: `agents/precision-tuning.md`

**Step 1: 创建 agent 文件**

```markdown
---
name: precision-tuning
version: 1.0.0
description: AscendC 算子精度调优 Agent — 修复编译通过但精度测试失败的 AscendC 算子
mode: subagent
temperature: 0.1

tools:
  write: true
  edit: true
  bash: true
  read: true

skills:
  - precision-tuning

argument-hint: >
  输入格式: "precision tune {task_name} [npu={NPU_ID}]"
  参数:
    - task_name: task 目录名（相对于 repo root，如 avg_pool3_d）
    - npu: NPU 设备 ID（默认 0）
  前提: task_name 目录下已有 model.py、model_new_ascendc.py、kernel/ 目录，
        且 evaluate_ascendc.sh 已报告 Numerical 失败（非 Build/Import 失败）。
---

# System Prompt

你是 **Precision Tuning Agent**，专门修复 AscendC 算子在编译通过后精度测试失败的问题。

## Role Definition

- **精度诊断专家**: 基于数值取证数据和代码分析，定位精度问题根因
- **精准修复者**: 根据诊断结果进行最小化、针对性的代码修复
- **流程遵守者**: 严格遵守 Gate 验证和循环控制信号

## Core Capabilities

（同 OpenOps 版本的精度分析能力，Phase A 参考文件已适配为 AscendOpGenAgent 资产）

## Operational Guidelines

参见 `skills/ascendc/precision-tuning/SKILL.md`。

### 工作目录限制

只允许读写 `{repo_root}/` 内的路径，包括：
- `{task_name}/` — 产物目录
- `skills/ascendc/` — 参考文件
- `archive_tasks/` — 参考案例
- `utils/` — 工具脚本

禁止访问父目录、绝对路径外位置，以及 `agents/`、`.claude/` 目录。

## Environment

CANN 8.1.rc1+, Ascend 910B。使用 Ascend C API (namespace AscendC)。

## Communication Style

- 所有思考、分析、推理使用中文
- English 仅用于：代码、技术标识符、JSON 键名、文件路径
```

**Step 2: 验证 agent 文件 frontmatter 格式（与现有 agents 对比）**
```bash
head -20 agents/ascend-kernel-developer.md
head -20 agents/precision-tuning.md
```
确认 frontmatter 格式一致。

**Step 3: Commit**
```bash
git add agents/precision-tuning.md
git commit -m "feat: add precision-tuning agent entry point"
```

---

## Task 7: 端到端冒烟测试

**Files:** 无新增，验证整体流程。

**Step 1: 找一个已有失败产物的 task（或构造一个）**

选择一个 `archive_tasks/` 中的已知通过案例，手动破坏其 kernel 代码（改一行计算），验证 agent 流程可以启动。

或直接检查路径约定是否正确：
```bash
# 验证 forensics 脚本能找到 verification_ascendc.py
python3 -c "
from pathlib import Path
script = Path('skills/ascendc/precision-tuning/scripts/precision_forensics.py')
import ast, sys
tree = ast.parse(script.read_text())
# 验证无旧路径引用
src = script.read_text()
for bad in ['_reference.py', '_custom.py', '_op_desc', 'OpName', 'op_kernel', 'op_host', 'generate_pybind', 'output_path']:
    if bad in src:
        print(f'WARNING: found old pattern: {bad}')
    else:
        print(f'OK: no {bad}')
"
```

**Step 2: 同样验证 gate.py**
```bash
python3 -c "
src = open('skills/ascendc/precision-tuning/scripts/precision_gate.py').read()
for bad in ['_find_project_dir', 'op_kernel', 'op_host', '_custom.cpp', 'output_path']:
    if bad in src:
        print(f'WARNING: found old pattern: {bad}')
    else:
        print(f'OK: no {bad}')
"
```

**Step 3: 验证 SKILL.md**
```bash
grep -c "dsl-lowering\|_reference\.py\|_custom\.py\|_op_desc\|OpName.*Custom\|op_kernel\|op_host\|generate_pybind\|evaluate\.py\b" \
    skills/ascendc/precision-tuning/SKILL.md
```
Expected: `0`

**Step 4: Final commit**
```bash
git add -A
git commit -m "feat: complete precision-tuning agent migration to AscendOpGenAgent"
```

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found | 15 findings, 0 fixed |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 0 | — | — |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

**CODEX:** 15 issues found in design doc — all addressed in revised findings.md before plan was written.
**VERDICT:** Codex review addressed. Eng review recommended before implementation.
