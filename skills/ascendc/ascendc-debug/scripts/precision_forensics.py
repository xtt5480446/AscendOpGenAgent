#!/usr/bin/env python3
"""
precision_forensics.py — 精度取证数值分析

职责边界:
  ✅ 做: 调用 verification_ascendc.py 获取数值差异数据、解析 stdout、
        pattern hint 分类、历史轮次对比
  ❌ 不做: 代码分析、根因诊断、修复建议（这些是 Agent 的事）

信息层级:
  L0: PASS/FAIL           — 直接从 verification_ascendc.py 返回码判定
  L1: 统计值              — basic_stats (from stdout)
  L2-L4: 解析自 stdout   — comparisons, mismatch_ratio, max_abs_diff
  L6: 内存布局            — inputs_meta (from stdout)
  L8: 算子语义            — 名称启发式推断算子类型

用法:
    python3 precision_forensics.py <task_name> [--attempt <N>] [--task-dir <path>]

输出:
    {task_dir}/precision_tuning/forensics_report_{attempt}.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR   = Path(__file__).resolve().parent          # scripts/
SKILL_DIR    = SCRIPT_DIR.parent                        # ascendc-debug/
REPO_ROOT    = SKILL_DIR.parent.parent.parent           # AscendOpGenAgent/
VERIF_SCRIPT = REPO_ROOT / "utils" / "verification_ascendc.py"


# ============================================================
# 算子类型检测 (L8 基础, 仅名称启发式)
# ============================================================

class OperatorTypeDetector:
    """基于算子名关键词推断算子类型，用于语义加权 pattern hint"""

    OP_TYPE_PATTERN_PRIORITY = {
        "reduction": ["magnitude_correlated", "tail_spike", "uniform_offset", "scattered"],
        "pooling": ["tail_spike", "boundary_concentration", "uniform_offset"],
        "loss": ["magnitude_correlated", "uniform_offset", "nan_inf_contamination"],
        "matmul": ["dimension_concentration", "scattered", "magnitude_correlated"],
        "activation": ["nan_inf_contamination", "uniform_offset", "boundary_concentration"],
        "normalization": ["nan_inf_contamination", "magnitude_correlated", "tail_spike"],
        "convolution": ["dimension_concentration", "boundary_concentration", "tail_spike"],
    }

    def detect(self, op_name: str, task_dir: str) -> dict:
        """基于算子名做关键词匹配推断算子类型。置信度低时标注 name_heuristic。

        返回 dict 含 SKILL.md Sub-step 2.1 要求的所有 L8_operator 字段:
          - op_type, source, confidence, pattern_priority (原有)
          - attributes: dict — 从 model.py 静态 AST 解析 Model.__init__ 提取
          - reduction_axis: {axis_index, axis_length} | None — 仅
            reduction/normalization/pooling 类型尝试推断
        """
        name = op_name.lower()
        type_map = [
            (["pool"], "pooling"),
            (["norm", "rms", "layer"], "normalization"),
            (["matmul", "gemm", "linear", "quant_matmul"], "matmul"),
            (["gather", "scatter", "index"], "gather"),
            (["concat", "cat"], "concat"),
            (["attn", "attention", "softmax"], "attention"),
            (["relu", "gelu", "silu", "activation", "elementwise"], "elementwise"),
            (["reduce", "sum", "mean", "max", "min", "prod", "cumsum"], "reduction"),
            (["loss", "mse", "cross_entropy", "bce"], "loss"),
            (["conv"], "convolution"),
        ]
        op_type = "unknown"
        for keywords, t in type_map:
            if any(kw in name for kw in keywords):
                op_type = t
                break

        attributes = self._extract_attributes(task_dir)
        reduction_axis = None
        if op_type in ("reduction", "normalization", "pooling"):
            reduction_axis = self._infer_reduction_axis(task_dir, attributes)

        return {
            "op_type": op_type,
            "source": "name_heuristic",
            "confidence": "low",
            "pattern_priority": self.OP_TYPE_PATTERN_PRIORITY.get(op_type, []),
            "attributes": attributes,
            "reduction_axis": reduction_axis,
        }

    # ------------------------------------------------------------------

    _ATTR_KEYS_OF_INTEREST = frozenset({
        "kernel_size", "stride", "padding", "dilation", "groups",
        "dim", "axis", "keepdim", "reduce",
        "normalized_shape", "eps", "elementwise_affine",
        "alpha", "beta", "gamma",
        "num_features", "num_classes",
        "in_features", "out_features", "in_channels", "out_channels",
        "scale", "scale_factor", "bias",
        "ceil_mode", "count_include_pad", "divisor_override",
        "return_indices", "output_size",
    })

    def _extract_attributes(self, task_dir: str) -> dict:
        """Static AST parse of model.py → attributes dict.

        Focuses on:
          - Model.__init__ default args (most common)
          - Top-level module constants (if Model refs them)
          - SCENARIOS / CASES lists (avg_pool3_d style — first entry's keys)

        Fails open: returns {} on any parse error. Never fatal.
        """
        import ast
        model_py = Path(task_dir) / "model.py"
        if not model_py.is_file():
            return {}
        try:
            source = model_py.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return {}

        attrs: dict = {}

        # 1) Model.__init__ default args
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name not in ("Model", "ModelNew"):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    defaults = args.defaults or []
                    named_args = args.args[-len(defaults):] if defaults else []
                    for arg, default in zip(named_args, defaults):
                        if arg.arg in self._ATTR_KEYS_OF_INTEREST:
                            try:
                                attrs[arg.arg] = ast.literal_eval(default)
                            except (ValueError, SyntaxError):
                                pass
            break

        # 2) SCENARIOS / CASES 列表 — 取第一个 entry 的 interested keys
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ("SCENARIOS", "CASES"):
                    if isinstance(node.value, ast.List) and node.value.elts:
                        first = node.value.elts[0]
                        if isinstance(first, ast.Dict):
                            for k_node, v_node in zip(first.keys, first.values):
                                if isinstance(k_node, ast.Constant) \
                                        and k_node.value in self._ATTR_KEYS_OF_INTEREST \
                                        and k_node.value not in attrs:
                                    try:
                                        attrs[k_node.value] = ast.literal_eval(v_node)
                                    except (ValueError, SyntaxError):
                                        pass
                    break

        return attrs

    def _infer_reduction_axis(self, task_dir: str, attrs: dict):
        """Infer {axis_index, axis_length} from attributes + first input group shape.

        Returns dict or None. axis_length may be None if input shape unavailable
        statically (caller fills from runtime tensor in Sub-step 2.1).
        """
        axis_index = attrs.get("dim", attrs.get("axis"))
        if axis_index is None:
            return None
        if not isinstance(axis_index, (int, list, tuple)):
            return None

        # Try to extract first input shape statically from SCENARIOS / CASES / explicit arrays.
        import ast
        model_py = Path(task_dir) / "model.py"
        if not model_py.is_file():
            return {"axis_index": axis_index, "axis_length": None}
        try:
            tree = ast.parse(model_py.read_text(encoding="utf-8"))
        except SyntaxError:
            return {"axis_index": axis_index, "axis_length": None}

        first_shape = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in ("SCENARIOS", "CASES"):
                    if isinstance(node.value, ast.List) and node.value.elts:
                        first = node.value.elts[0]
                        if isinstance(first, ast.Dict):
                            for k_node, v_node in zip(first.keys, first.values):
                                if isinstance(k_node, ast.Constant) and k_node.value == "shape":
                                    try:
                                        first_shape = tuple(ast.literal_eval(v_node))
                                    except (ValueError, SyntaxError):
                                        pass
                                    break
                    break

        if first_shape is None:
            return {"axis_index": axis_index, "axis_length": None}

        try:
            if isinstance(axis_index, int):
                ai = axis_index if axis_index >= 0 else len(first_shape) + axis_index
                if 0 <= ai < len(first_shape):
                    return {"axis_index": axis_index, "axis_length": int(first_shape[ai])}
        except Exception:
            pass
        return {"axis_index": axis_index, "axis_length": None}


# ============================================================
# 执行引擎 — spawn _forensics_child.py 子进程, 读回 pickle dumped tensors
# ============================================================

import pickle
import shutil
import tempfile

CHILD_SCRIPT = SCRIPT_DIR / "_forensics_child.py"


class OperatorExecutor:
    """Spawn _forensics_child.py subprocess. Child loads model.py +
    model_new_ascendc.py, runs all input_groups, pickle-dumps per-case
    tensors to a tmp dir. Parent reads them back for DiffAnalyzer.

    与 utils/verification_ascendc.py 子进程隔离, 避免主进程 sys.path /
    torch state / extension 污染; child 的 loader 是 verification_ascendc.py
    的副本 (见 _forensics_child.py 顶部说明) + parity 测试保证语义一致。
    """

    SUBPROCESS_TIMEOUT_SEC = 1800  # 30 min

    def __init__(self, op_name: str, task_dir: str, attempt: int = 0):
        self.op_name = op_name
        self.task_dir = str(Path(task_dir).resolve())
        self.attempt = attempt

    def load_and_execute(self) -> dict:
        """Returns:
          {
            "cases":    [{"case_idx", "ref", "cand", "inputs",
                          "int8_triggered", "atol", "rtol"}, ...],
            "metadata": {"status", "num_cases", "device",
                         "atol_default", "rtol_default",
                         "int8_atol", "int8_rtol",
                         "loader_parity_hash", ...},
          }

        Raises RuntimeError on child subprocess failure.
        """
        # 使用 task_dir 内的临时目录; 主进程读完 cases 后清理
        dump_root = Path(self.task_dir) / "precision_tuning" / f".forensics_tmp_{self.attempt}"
        if dump_root.exists():
            shutil.rmtree(dump_root)
        dump_root.mkdir(parents=True, exist_ok=True)

        env = {**os.environ}
        # 保留 caller 已设定的 ASCEND_RT_VISIBLE_DEVICES (不覆盖), 与 bench 一致

        try:
            proc = subprocess.run(
                [sys.executable, str(CHILD_SCRIPT), self.task_dir, str(dump_root)],
                capture_output=True, text=True, env=env,
                timeout=self.SUBPROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"forensics child timed out after {self.SUBPROCESS_TIMEOUT_SEC}s"
            ) from e

        meta_path = dump_root / "metadata.json"
        if not meta_path.is_file():
            raise RuntimeError(
                f"forensics child produced no metadata.json (rc={proc.returncode})\n"
                f"stderr: {proc.stderr[-2000:]}"
            )
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"forensics child metadata.json not parseable: {e}") from e

        if metadata.get("status") != "completed" or proc.returncode != 0:
            raise RuntimeError(
                f"forensics child failed (rc={proc.returncode}, "
                f"status={metadata.get('status')}): "
                f"{metadata.get('error', '<no error>')}\n"
                f"stderr tail: {proc.stderr[-1500:]}"
            )

        cases = []
        for i in range(int(metadata.get("num_cases", 0))):
            case_path = dump_root / f"case_{i}.pkl"
            if not case_path.is_file():
                raise RuntimeError(f"forensics child missing case_{i}.pkl")
            with case_path.open("rb") as f:
                cases.append(pickle.load(f))

        return {"cases": cases, "metadata": metadata, "dump_dir": str(dump_root)}

    def cleanup_dump(self, dump_dir: str) -> None:
        """Remove child's tmp dump dir after analysis. Safe to call twice."""
        try:
            shutil.rmtree(dump_dir, ignore_errors=True)
        except Exception:
            pass


# ============================================================
# L5: 中间结果探测 (TODO - Phase 2)
# ============================================================

class IntermediateProbe:
    def probe(self, op_name: str, task_dir: str) -> dict | None:
        return None


# ============================================================
# L7: 代码位置映射 (TODO - Phase 2 / Agent)
# ============================================================

class CodeMapper:
    def map(self, worst_elements: list, kernel_path: str) -> dict | None:
        return None


# ============================================================
# 历史轮次对比
# ============================================================

class HistoryComparator:

    def __init__(self, tuning_dir: str, current_attempt: int):
        self.tuning_dir = tuning_dir
        self.current_attempt = current_attempt

    def load_history(self) -> list:
        history = []
        for i in range(self.current_attempt):
            p = os.path.join(self.tuning_dir, "history", f"attempt_{i}", "forensics_report.json")
            if os.path.exists(p):
                with open(p) as f:
                    history.append({"attempt": i, "report": json.load(f)})
        return history

    def build_trend(self, current_report: dict) -> dict | None:
        history = self.load_history()
        if not history:
            return None
        trend = []
        for h in history:
            r = h["report"]
            if r.get("status") != "completed" or not r.get("outputs"):
                continue
            s = r["outputs"][0].get("basic_stats", {})
            trend.append({
                "attempt": h["attempt"],
                "mismatch_ratio": s.get("mismatch_ratio"),
                "max_abs_diff": s.get("max_abs_diff"),
                "primary_hint": r.get("primary_hint"),
            })
        if current_report.get("outputs"):
            cs = current_report["outputs"][0].get("basic_stats", {})
            trend.append({
                "attempt": self.current_attempt,
                "mismatch_ratio": cs.get("mismatch_ratio"),
                "max_abs_diff": cs.get("max_abs_diff"),
                "primary_hint": current_report.get("primary_hint"),
            })
        return {
            "num_attempts": len(trend),
            "trend": trend,
            "mismatch_improving": self._improving(trend),
        }

    def _improving(self, trend):
        ratios = [t["mismatch_ratio"] for t in trend if t["mismatch_ratio"] is not None]
        return ratios[-1] < ratios[-2] if len(ratios) >= 2 else True


# ============================================================
# L1-L4 数值 Diff 分析
# ============================================================
#
# 语义边界（与 utils/verification_ascendc.py 对齐）:
#   - ATOL/RTOL 与 verification 保持一致 (atol=1e-2, rtol=1e-2); int8 特判时切
#     atol=1.5, rtol=0.0（由 caller 通过 analyzer.ATOL/analyzer.RTOL 覆盖）。
#   - mismatch mask = abs(golden - actual) > ATOL + RTOL * abs(golden)。
#   - 本模块不做 torch.nan_to_num 替换；caller 若要与 verification pass/fail 一致，
#     应在传入前对 golden/actual 做 nan_to_num(nan=0, posinf=1e9, neginf=-1e9)。
#   - value_range 里仍记录原始 NaN/Inf 计数（caller 需传原始 tensor 的 numpy 视图
#     做 value_range 分析，或在 pre-nan-to-num 快照上额外跑）。
#
# 所有方法接收 np.ndarray (CPU, float32 upcast 后)，不依赖 torch / torch_npu。

class DiffAnalyzer:

    ATOL = 1e-02
    RTOL = 1e-02

    def __init__(self, op_type_info: dict = None):
        self.op_type_info = op_type_info or {"op_type": "unknown", "pattern_priority": []}

    def analyze(self, golden: np.ndarray, actual: np.ndarray) -> dict:
        abs_diff = np.abs(golden - actual)
        threshold = self.ATOL + self.RTOL * np.abs(golden)
        mismatch_mask = abs_diff > threshold

        return {
            "pass_fail": bool(np.sum(mismatch_mask) == 0),
            "basic_stats": self._basic_stats(golden, actual, abs_diff, mismatch_mask),
            "error_distribution": self._error_distribution(golden, actual, abs_diff),
            "value_range": self._value_range(golden, actual),
            "pattern_hint": self._classify_pattern(golden, actual, abs_diff, mismatch_mask),
            "worst_elements": self._worst_elements(abs_diff, golden, actual, top_k=10),
            "tail_analysis": self._tail_analysis(abs_diff, mismatch_mask, golden.shape),
            "dimension_analysis": self._dimension_analysis(abs_diff, mismatch_mask, golden.shape),
            "L5_intermediate": None,
            "L7_code_mapping": None,
            "L8_op_type": self.op_type_info.get("op_type", "unknown"),
        }

    # ---- L1 ----

    def _basic_stats(self, golden, actual, abs_diff, mismatch_mask) -> dict:
        total = max(golden.size, 1)
        n = int(np.sum(mismatch_mask))
        return {
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "median_abs_diff": float(np.median(abs_diff)),
            "p99_abs_diff": float(np.percentile(abs_diff, 99)),
            "num_mismatched": n,
            "total_elements": int(golden.size),
            "mismatch_ratio": n / total,
            "match_rate": 1.0 - n / total,
            "int8_special_tolerance": False,   # caller 在 int8 路径下设为 True
        }

    def _error_distribution(self, golden, actual, abs_diff) -> dict:
        diff_signed = (actual - golden).flatten()
        abs_flat = abs_diff.flatten()
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        quartile = {f"p{p}": float(np.percentile(abs_flat, p)) for p in percentiles}

        golden_flat = golden.flatten()
        safe = np.abs(golden_flat) > 1e-7
        rel = np.zeros_like(abs_flat)
        if np.any(safe):
            rel[safe] = abs_flat[safe] / np.abs(golden_flat[safe])

        n_pos = int(np.sum(diff_signed > 0))
        n_neg = int(np.sum(diff_signed < 0))
        return {
            "abs_diff_percentiles": quartile,
            "rel_error_mean": float(np.mean(rel)),
            "rel_error_max": float(min(np.max(rel), 1e6)),
            "sign_analysis": {
                "positive_count": n_pos,
                "negative_count": n_neg,
                "zero_count": int(np.sum(diff_signed == 0)),
                "bias_direction": "positive" if n_pos > n_neg * 1.5
                                  else "negative" if n_neg > n_pos * 1.5
                                  else "balanced",
                "mean_signed_diff": float(np.mean(diff_signed)),
            },
        }

    def _value_range(self, golden, actual) -> dict:
        def _s(arr, name):
            return {
                f"{name}_min": float(np.min(arr)), f"{name}_max": float(np.max(arr)),
                f"{name}_mean": float(np.mean(arr)), f"{name}_std": float(np.std(arr)),
                f"{name}_has_nan": bool(np.any(np.isnan(arr))),
                f"{name}_has_inf": bool(np.any(np.isinf(arr))),
                f"{name}_nan_count": int(np.sum(np.isnan(arr))),
                f"{name}_inf_count": int(np.sum(np.isinf(arr))),
            }
        r = {}
        r.update(_s(golden, "golden"))
        r.update(_s(actual, "actual"))
        return r

    # ---- L2/L3: Pattern Hint (含语义加权) ----

    def _classify_pattern(self, golden, actual, abs_diff, mismatch_mask) -> dict:
        shape = golden.shape
        total = max(golden.size, 1)
        mismatch_ratio = np.sum(mismatch_mask) / total
        hints = []

        nan_n = int(np.sum(np.isnan(actual)))
        inf_n = int(np.sum(np.isinf(actual)))
        if nan_n > 0 or inf_n > 0:
            hints.append({"pattern": "nan_inf_contamination", "confidence": 0.95,
                          "evidence": f"NPU 输出含 NaN={nan_n}, Inf={inf_n}"})

        if mismatch_ratio > 0.9:
            dv = (actual - golden).flatten()
            dm, ds = float(np.mean(dv)), float(np.std(dv))
            if ds < 0.1 * abs(dm) and abs(dm) > 1e-3:
                hints.append({"pattern": "uniform_offset", "confidence": 0.85,
                              "evidence": f"全局偏移 mean={dm:.6f}, std={ds:.6f}"})
            else:
                hints.append({"pattern": "all_wrong", "confidence": 0.9,
                              "evidence": f"mismatch={mismatch_ratio:.1%}"})

        t = self._check_tail_spike(mismatch_mask, shape)
        if t:
            hints.append(t)
        if len(shape) >= 2:
            c = self._check_dim_concentration(mismatch_mask, shape)
            if c:
                hints.append(c)
        m = self._check_magnitude_correlation(golden, mismatch_mask)
        if m:
            hints.append(m)
        b = self._check_boundary_concentration(mismatch_mask, shape)
        if b:
            hints.append(b)

        if not hints:
            hints.append({"pattern": "scattered", "confidence": 0.4,
                          "evidence": f"mismatch 分散, 比例={mismatch_ratio:.2%}"})

        # 语义加权：op_type 的 pattern_priority 里靠前的 pattern confidence 抬升
        plist = self.op_type_info.get("pattern_priority", [])
        if plist:
            for h in hints:
                if h["pattern"] in plist:
                    rank = plist.index(h["pattern"])
                    boost = max(0, 0.1 - rank * 0.03)
                    h["confidence"] = min(0.99, h["confidence"] + boost)
                    h["semantic_boosted"] = True

        hints.sort(key=lambda h: h["confidence"], reverse=True)
        return {
            "primary_hint": hints[0]["pattern"],
            "primary_confidence": hints[0]["confidence"],
            "primary_evidence": hints[0]["evidence"],
            "all_hints": hints,
        }

    def _check_tail_spike(self, mm, shape):
        if len(shape) == 0:
            return None
        ld = shape[-1]
        for ts in [8, 16, 32, 64, 128, 256]:
            if ld <= ts:
                continue
            tl = ld % ts
            if tl == 0:
                continue
            t_s = tuple([slice(None)] * (len(shape) - 1) + [slice(-tl, None)])
            b_s = tuple([slice(None)] * (len(shape) - 1) + [slice(0, -tl)])
            tr = float(np.mean(mm[t_s]))
            br = float(np.mean(mm[b_s]))
            if tr > 0.3 and (br < 0.01 or tr > br * 5):
                return {"pattern": "tail_spike",
                        "confidence": min(0.9, 0.6 + (tr - br)),
                        "evidence": f"last_dim={ld}, tile={ts}, 尾块({tl})={tr:.1%}, 主体={br:.1%}",
                        "detail": {"tile_size": ts, "tail_len": tl,
                                   "tail_rate": tr, "body_rate": br}}
        return None

    def _check_dim_concentration(self, mm, shape):
        for dim in range(len(shape)):
            if shape[dim] <= 1:
                continue
            rates = [float(np.mean(mm[tuple([slice(None)] * dim + [i]
                                           + [slice(None)] * (len(shape) - dim - 1))]))
                     for i in range(shape[dim])]
            ra = np.array(rates)
            if np.max(ra) > 0.5 and np.min(ra) < 0.1:
                bad = [int(i) for i in np.where(ra > 0.3)[0]]
                return {"pattern": "dimension_concentration", "confidence": 0.8,
                        "evidence": f"dim={dim}(size={shape[dim]}), 索引 {bad} mismatch 偏高",
                        "detail": {"dim": dim, "bad_indices": bad,
                                   "rates": [round(r, 4) for r in rates]}}
        return None

    def _check_magnitude_correlation(self, golden, mm):
        gf, mf = np.abs(golden.flatten()), mm.flatten()
        if np.sum(mf) < 10 or np.sum(~mf) < 10:
            return None
        mmv, nmv = float(np.mean(gf[mf])), float(np.mean(gf[~mf]))
        if nmv < 1e-10:
            return None
        ratio = mmv / nmv
        if ratio > 3.0:
            return {"pattern": "magnitude_correlated",
                    "confidence": min(0.85, 0.5 + (ratio - 3) * 0.05),
                    "evidence": f"mismatch 均值({mmv:.4f})是正常({nmv:.4f})的{ratio:.1f}倍"}
        if ratio < 0.3:
            return {"pattern": "magnitude_correlated",
                    "confidence": min(0.85, 0.5 + (1 / ratio - 3) * 0.05),
                    "evidence": f"mismatch 集中在小值区域"}
        return None

    def _check_boundary_concentration(self, mm, shape):
        if len(shape) < 2:
            return None
        tm = int(np.sum(mm))
        if tm == 0:
            return None
        bm, bt = 0, 0
        for dim in range(len(shape)):
            if shape[dim] <= 2:
                continue
            for edge in [0, shape[dim] - 1]:
                s = tuple([slice(None)] * dim + [edge]
                          + [slice(None)] * (len(shape) - dim - 1))
                bm += int(np.sum(mm[s]))
                bt += mm[s].size
        if bt > 0 and tm > 0 and bm / tm > 0.6 and bm / bt > 0.2:
            return {"pattern": "boundary_concentration", "confidence": 0.75,
                    "evidence": f"边界含 {bm/tm:.0%} 的 mismatch"}
        return None

    # ---- L4 ----

    def _worst_elements(self, abs_diff, golden, actual, top_k=10):
        flat = abs_diff.flatten()
        k = min(top_k, len(flat))
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        return [{"index": list(map(int, np.unravel_index(i, abs_diff.shape))),
                 "abs_diff": float(flat[i]),
                 "golden_value": float(golden.flat[i]),
                 "actual_value": float(actual.flat[i]),
                 "L7_gm_offset": None, "L7_source_line": None} for i in idx]

    def _tail_analysis(self, abs_diff, mm, shape):
        if len(shape) == 0:
            return {"last_dim": None, "note": "scalar output has no tail dimension"}
        ld = shape[-1]
        results = {}
        for ts in [8, 16, 32, 64, 128, 256]:
            tl = ld % ts
            if tl == 0 or ld <= ts:
                continue
            t_s = tuple([slice(None)] * (len(shape) - 1) + [slice(-tl, None)])
            b_s = tuple([slice(None)] * (len(shape) - 1) + [slice(0, -tl)])
            results[f"tile_{ts}"] = {
                "tail_len": tl,
                "tail_mean_diff": float(np.mean(abs_diff[t_s])),
                "body_mean_diff": float(np.mean(abs_diff[b_s])),
                "tail_max_diff": float(np.max(abs_diff[t_s])),
                "body_max_diff": float(np.max(abs_diff[b_s])),
                "tail_mismatch_rate": float(np.mean(mm[t_s])),
                "body_mismatch_rate": float(np.mean(mm[b_s])),
            }
        if not results:
            return {"last_dim": ld, "note": "last_dim 是常见 tile 的整数倍"}
        results["last_dim"] = ld
        return results

    def _dimension_analysis(self, abs_diff, mm, shape):
        analysis = []
        for dim in range(len(shape)):
            if shape[dim] <= 1:
                continue
            rates, diffs = [], []
            for i in range(shape[dim]):
                s = tuple([slice(None)] * dim + [i]
                          + [slice(None)] * (len(shape) - dim - 1))
                rates.append(float(np.mean(mm[s])))
                diffs.append(float(np.mean(abs_diff[s])))
            analysis.append({
                "dim": dim, "size": shape[dim],
                "mismatch_rate_min": float(np.min(rates)),
                "mismatch_rate_max": float(np.max(rates)),
                "mismatch_rate_std": float(np.std(rates)),
                "mean_diff_min": float(np.min(diffs)),
                "mean_diff_max": float(np.max(diffs)),
                "per_index_rates": [round(r, 4) for r in rates] if shape[dim] <= 64 else None,
            })
        return analysis


# ============================================================
# L6: 内存布局分析
# ============================================================
#
# 不触发 NPU 操作，只读 tensor.shape/stride/dtype/是否连续/最后一维对齐情况。
# 用于 Sub-step 2.1 [FORENSICS_SUMMARY] 中 L6 内存布局一节。

class MemoryLayoutAnalyzer:

    TILE_SIZES = [8, 16, 32, 64, 128, 256]

    def analyze_tensors(self, tensors: list, label: str = "input") -> list:
        results = []
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                continue
            results.append(self._analyze_single(t, f"{label}_{i}"))
        return results

    def _analyze_single(self, t: torch.Tensor, name: str) -> dict:
        info = {
            "name": name, "shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "is_contiguous": t.is_contiguous(),
            "storage_offset": t.storage_offset(),
            "element_size_bytes": t.element_size(),
        }
        last_dim = t.shape[-1] if t.ndim > 0 else 0
        info["last_dim_alignment"] = {
            f"tile_{ts}": {"remainder": last_dim % ts, "aligned": last_dim % ts == 0}
            for ts in self.TILE_SIZES
        }
        return info


# ============================================================
# Output 扁平化 (对齐 utils/verification_ascendc.py::_compare_values 的 path 约定)
# ============================================================
#
# verification_ascendc.py 的 _compare_values 递归处理任意
# Tensor | list | tuple | dict | scalar 结构, path 语法：
#   - "output[0]"         : 顶层 tensor / 列表索引
#   - "output[1].foo"     : dict key
#   - "output[0][2]"      : 嵌套 list/tuple 索引
#
# OutputFlattener.flatten(ref, cand, root="output") 把 (ref, cand) 这两棵
# 同形状的嵌套树展开成 {path_str: {ref, cand, kind, shape, dtype, status}}。
#
# status 值：
#   - "ok"              : 结构/类型匹配，可进 DiffAnalyzer
#   - "type_mismatch"   : ref/cand 类型不同
#   - "shape_mismatch"  : tensor shape 不同
#   - "len_mismatch"    : list/tuple 长度不同
#   - "key_mismatch"    : dict key 集合不同
#
# kind 值："tensor" | "scalar" | "none"
#
# 若 status != "ok"，DiffAnalyzer 不应被调用（caller 要先检查）。

class OutputFlattener:

    def flatten(self, ref, cand, root: str = "output") -> dict:
        out: dict = {}
        self._walk(ref, cand, root, out)
        return out

    def _walk(self, ref, cand, path: str, out: dict) -> None:
        # --- type mismatch 顶层优先判定 ---
        if type(ref) is not type(cand):
            # 除了 list/tuple 互通这种 Python 内置差异，其他都视作 type_mismatch
            if not (isinstance(ref, (list, tuple)) and isinstance(cand, (list, tuple))
                    and type(ref) is type(cand)):
                out[path] = {
                    "ref": ref, "cand": cand, "kind": "none", "shape": None,
                    "dtype": None, "status": "type_mismatch",
                }
                return

        # --- Tensor ---
        if isinstance(ref, torch.Tensor):
            shape = list(ref.shape)
            dtype = str(ref.dtype)
            if ref.shape != cand.shape:
                out[path] = {
                    "ref": ref, "cand": cand, "kind": "tensor", "shape": shape,
                    "dtype": dtype, "status": "shape_mismatch",
                }
                return
            out[path] = {
                "ref": ref, "cand": cand, "kind": "tensor", "shape": shape,
                "dtype": dtype, "status": "ok",
            }
            return

        # --- list / tuple ---
        if isinstance(ref, (list, tuple)):
            if len(ref) != len(cand):
                out[path] = {
                    "ref": ref, "cand": cand, "kind": "none", "shape": None,
                    "dtype": None, "status": "len_mismatch",
                }
                return
            for i, (r, c) in enumerate(zip(ref, cand)):
                self._walk(r, c, f"{path}[{i}]", out)
            return

        # --- dict ---
        if isinstance(ref, dict):
            if set(ref.keys()) != set(cand.keys()):
                out[path] = {
                    "ref": ref, "cand": cand, "kind": "none", "shape": None,
                    "dtype": None, "status": "key_mismatch",
                }
                return
            for k in ref:
                self._walk(ref[k], cand[k], f"{path}.{k}", out)
            return

        # --- None ---
        if ref is None:
            out[path] = {
                "ref": None, "cand": None, "kind": "none", "shape": None,
                "dtype": None, "status": "ok",
            }
            return

        # --- scalar ---
        out[path] = {
            "ref": ref, "cand": cand, "kind": "scalar", "shape": None,
            "dtype": type(ref).__name__, "status": "ok",
        }


# ============================================================
# 主入口
# ============================================================

class PrecisionForensics:

    def __init__(self, op_name: str, task_dir: str, attempt: int = 0):
        self.op_name = op_name
        self.task_dir = task_dir
        self.attempt = attempt
        self.tuning_dir = os.path.join(task_dir, "precision_tuning")
        os.makedirs(self.tuning_dir, exist_ok=True)

    def run(self) -> dict:
        executor = OperatorExecutor(self.op_name, self.task_dir, self.attempt)
        dump_dir = None
        try:
            op_type_info = OperatorTypeDetector().detect(self.op_name, self.task_dir)
            data = executor.load_and_execute()
            cases = data["cases"]
            metadata = data["metadata"]
            dump_dir = data["dump_dir"]
            num_cases = len(cases)

            analyzer = DiffAnalyzer(op_type_info=op_type_info)
            layout = MemoryLayoutAnalyzer()
            flattener = OutputFlattener()

            # Flatten each case → {path: {ref, cand, kind, shape, dtype, status}}
            per_case_flat = [
                flattener.flatten(c["ref"], c["cand"], root="output") for c in cases
            ]
            all_paths = sorted(set().union(*[set(p.keys()) for p in per_case_flat])) \
                if per_case_flat else []

            output_reports = []
            int8_any = False
            nan_inf_agg = self._init_nan_inf_agg()

            for output_index, path in enumerate(all_paths):
                per_case_diffs = self._analyze_path_across_cases(
                    path, per_case_flat, cases, analyzer
                )
                if not per_case_diffs:
                    continue  # all cases had non-ok status at this path — skip
                for pcd in per_case_diffs:
                    if pcd.get("_int8"):
                        int8_any = True
                    self._accumulate_nan_inf(nan_inf_agg, pcd)

                rep_idx = self._pick_representative(per_case_diffs)
                rep = per_case_diffs[rep_idx]
                agg = self._compute_case_aggregate(per_case_diffs, cases)
                first_pcp = next(
                    (p[path] for p in per_case_flat if path in p), {}
                )
                output_reports.append({
                    "output_index": output_index,
                    "output_path": path,
                    "output_kind": first_pcp.get("kind", "unknown"),
                    "output_shape": first_pcp.get("shape"),
                    "output_dtype": first_pcp.get("dtype"),
                    "pass_fail": all(
                        d["basic_stats"]["num_mismatched"] == 0 for d in per_case_diffs
                    ),
                    # pattern_hint & diagnostic fields come from representative case
                    "basic_stats": rep["basic_stats"],
                    "error_distribution": rep["error_distribution"],
                    "value_range": rep["value_range"],
                    "pattern_hint": rep["pattern_hint"],
                    "worst_elements": rep["worst_elements"],
                    "tail_analysis": rep["tail_analysis"],
                    "dimension_analysis": rep["dimension_analysis"],
                    "L5_intermediate": None,
                    "L7_code_mapping": None,
                    "L8_op_type": op_type_info.get("op_type", "unknown"),
                    # cross-case payload (codex HIGH: do NOT discard non-worst cases)
                    "per_case": [self._strip_private_keys(d) for d in per_case_diffs],
                    "representative_case_idx": per_case_diffs[rep_idx]["case_idx"],
                    "case_aggregate": agg,
                })

            # Fallback: if no tensor outputs were analyzable, produce a
            # synthetic pass/fail report from metadata to keep Gate-F schema happy.
            if not output_reports:
                output_reports.append(self._synthetic_report(cases, metadata, op_type_info))

            # primary_hint from worst output (codex HIGH修订)
            worst_output = max(
                output_reports,
                key=lambda o: (
                    (o.get("case_aggregate") or {}).get("mismatch_ratio_max",
                        o["basic_stats"]["mismatch_ratio"]),
                    (o.get("case_aggregate") or {}).get("max_abs_diff_max",
                        o["basic_stats"]["max_abs_diff"]),
                ),
            )
            ph = worst_output["pattern_hint"]

            # Top-level L6_memory_layout.inputs: layout of representative case's inputs
            if cases:
                rep_case_inputs = cases[worst_output.get(
                    "representative_case_idx", 0)]["inputs"] \
                    if isinstance(worst_output.get("representative_case_idx"), int) \
                    else cases[0]["inputs"]
                inputs_layout = self._extract_input_tensors_for_layout(rep_case_inputs)
                inputs_layout_report = layout.analyze_tensors(inputs_layout, "input")
            else:
                inputs_layout_report = []

            comparator = HistoryComparator(self.tuning_dir, self.attempt)

            final = {
                "version": "2.0",
                "op_name": self.op_name,
                "attempt": self.attempt,
                "status": "completed",
                "num_outputs": len(output_reports),
                "L0_pass": all(o["pass_fail"] for o in output_reports),
                "all_passed": all(o["pass_fail"] for o in output_reports),
                "outputs": output_reports,
                "L5_intermediate": None,
                "L6_memory_layout": {
                    "inputs": inputs_layout_report,
                    "outputs": [],   # per-output shape lives in outputs[i].output_shape
                },
                "L7_code_mapping": None,
                "L8_operator": op_type_info,
                "primary_hint": ph["primary_hint"],
                "primary_confidence": ph["primary_confidence"],
                "primary_evidence": ph["primary_evidence"],
                "all_hints": ph.get("all_hints", []),
                "history_trend": None,
                "multi_case_analysis": None,
                "num_test_cases": num_cases,
                "available_files": {
                    "reference": os.path.exists(os.path.join(self.task_dir, "model.py")),
                    "custom": os.path.exists(
                        os.path.join(self.task_dir, "model_new_ascendc.py")
                    ),
                },
                "int8_path_active": int8_any,
                "nan_inf_detected": nan_inf_agg,
                "executor_parity_hash": metadata.get("loader_parity_hash"),
            }

            trend = comparator.build_trend(final)
            if trend:
                final["history_trend"] = trend

            report_path = os.path.join(
                self.tuning_dir, f"forensics_report_{self.attempt}.json"
            )
            with open(report_path, "w") as f:
                json.dump(final, f, indent=2, ensure_ascii=False)

            stats = worst_output["basic_stats"]
            print(f"[FORENSICS] ✅ 精度取证完成 (attempt={self.attempt})")
            print(f"  op_type: {op_type_info['op_type']} "
                  f"(source={op_type_info.get('source')})")
            print(f"  primary_hint: {final['primary_hint']} "
                  f"(confidence={final['primary_confidence']:.2f})")
            print(f"  evidence: {final['primary_evidence']}")
            print(f"  mismatch: {stats['mismatch_ratio']:.2%} "
                  f"({stats['num_mismatched']}/{stats['total_elements']})")
            print(f"  max_diff: {stats['max_abs_diff']:.6f}")
            print(f"  num_outputs: {final['num_outputs']}, "
                  f"num_cases: {num_cases}, "
                  f"int8_active: {int8_any}")
            if trend:
                print(f"  趋势: {'↓ 改善中' if trend.get('mismatch_improving') else '↑ 未改善'}")
            print(f"  report: {report_path}")
            return final

        except Exception as e:
            err = {
                "version": "2.0", "op_name": self.op_name, "attempt": self.attempt,
                "status": "error", "error": str(e), "traceback": traceback.format_exc(),
                "outputs": [], "primary_hint": "error",
            }
            rp = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
            with open(rp, "w") as f:
                json.dump(err, f, indent=2, ensure_ascii=False)
            print(f"[FORENSICS] ❌ 取证失败: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        finally:
            if dump_dir:
                executor.cleanup_dump(dump_dir)

    # ------------------------------------------------------------------
    # Helpers for new run()

    def _analyze_path_across_cases(
        self, path: str, per_case_flat: list, cases: list, analyzer
    ) -> list:
        """For each case that has an 'ok' tensor at `path`, run DiffAnalyzer.

        Returns list of per-case diff dicts, each tagged with case_idx and
        a private `_int8` / `_nan_inf_src` flag for aggregation.
        """
        per_case_diffs = []
        for case_idx, flat in enumerate(per_case_flat):
            if path not in flat:
                continue
            pcp = flat[path]
            if pcp["kind"] != "tensor" or pcp["status"] != "ok":
                continue

            case = cases[case_idx]
            atol = float(case["atol"])
            rtol = float(case["rtol"])
            is_int8 = bool(case["int8_triggered"])

            ref_t = pcp["ref"]
            cand_t = pcp["cand"]

            # Record raw NaN/Inf on pre-nan_to_num tensors (for diagnostics)
            nan_inf_ref = self._nan_inf_counts(ref_t)
            nan_inf_cand = self._nan_inf_counts(cand_t)

            # Comparison mask follows verification_ascendc: nan_to_num, then
            # abs-diff > atol + rtol * abs(golden/cand).
            ref_fp = torch.nan_to_num(ref_t.to(torch.float32))
            cand_fp = torch.nan_to_num(cand_t.to(torch.float32))
            ref_np = ref_fp.cpu().numpy()
            cand_np = cand_fp.cpu().numpy()

            analyzer.ATOL = atol
            analyzer.RTOL = rtol
            diff = analyzer.analyze(ref_np, cand_np)
            diff["basic_stats"]["int8_special_tolerance"] = is_int8
            diff["case_idx"] = case_idx
            diff["_int8"] = is_int8
            diff["_nan_inf_src"] = {"ref": nan_inf_ref, "cand": nan_inf_cand}
            per_case_diffs.append(diff)
        return per_case_diffs

    def _pick_representative(self, per_case_diffs: list) -> int:
        """max mismatch_ratio; tiebreak max_abs_diff."""
        return max(
            range(len(per_case_diffs)),
            key=lambda i: (
                per_case_diffs[i]["basic_stats"]["mismatch_ratio"],
                per_case_diffs[i]["basic_stats"]["max_abs_diff"],
            ),
        )

    def _compute_case_aggregate(self, per_case_diffs: list, cases: list) -> dict:
        ratios = [d["basic_stats"]["mismatch_ratio"] for d in per_case_diffs]
        max_diffs = [d["basic_stats"]["max_abs_diff"] for d in per_case_diffs]
        n_pass = sum(
            1 for d in per_case_diffs if d["basic_stats"]["num_mismatched"] == 0
        )
        n_fail = len(per_case_diffs) - n_pass

        # heuristic: shape_conditional — does mismatch ratio correlate with
        # "last dim" of the case's first input tensor?
        shape_conditional = False
        try:
            last_dims = []
            for d in per_case_diffs:
                inputs = cases[d["case_idx"]]["inputs"]
                first_tensor = self._first_tensor_in(inputs)
                if first_tensor is not None and first_tensor.ndim > 0:
                    last_dims.append(int(first_tensor.shape[-1]))
                else:
                    last_dims.append(0)
            if len(last_dims) >= 3 and len(set(last_dims)) >= 2:
                r_last = np.array(last_dims, dtype=float)
                r_mis = np.array(ratios, dtype=float)
                if r_last.std() > 0 and r_mis.std() > 0:
                    corr = float(np.corrcoef(r_last, r_mis)[0, 1])
                    if not np.isnan(corr) and abs(corr) > 0.7:
                        shape_conditional = True
        except Exception:
            pass

        all_same_pattern = (
            len({d["pattern_hint"]["primary_hint"] for d in per_case_diffs}) == 1
        )

        return {
            "num_cases": len(per_case_diffs),
            "mismatch_ratio_min": min(ratios),
            "mismatch_ratio_max": max(ratios),
            "mismatch_ratio_mean": sum(ratios) / len(ratios),
            "max_abs_diff_min": min(max_diffs),
            "max_abs_diff_max": max(max_diffs),
            "pass_case_count": n_pass,
            "fail_case_count": n_fail,
            "all_cases_same_pattern": all_same_pattern,
            "shape_conditional": shape_conditional,
        }

    def _init_nan_inf_agg(self) -> dict:
        return {
            "ref": {"has_nan": False, "has_inf": False,
                    "nan_count": 0, "inf_count": 0},
            "cand": {"has_nan": False, "has_inf": False,
                     "nan_count": 0, "inf_count": 0},
        }

    def _accumulate_nan_inf(self, agg: dict, pcd: dict) -> None:
        src = pcd.get("_nan_inf_src") or {}
        for side in ("ref", "cand"):
            s = src.get(side, {})
            agg[side]["has_nan"] = agg[side]["has_nan"] or bool(s.get("has_nan"))
            agg[side]["has_inf"] = agg[side]["has_inf"] or bool(s.get("has_inf"))
            agg[side]["nan_count"] += int(s.get("nan_count", 0))
            agg[side]["inf_count"] += int(s.get("inf_count", 0))

    def _nan_inf_counts(self, t) -> dict:
        if not isinstance(t, torch.Tensor):
            return {"has_nan": False, "has_inf": False, "nan_count": 0, "inf_count": 0}
        nan_n = int(torch.isnan(t).sum().item()) if t.numel() else 0
        inf_n = int(torch.isinf(t).sum().item()) if t.numel() else 0
        return {
            "has_nan": nan_n > 0, "has_inf": inf_n > 0,
            "nan_count": nan_n, "inf_count": inf_n,
        }

    def _strip_private_keys(self, d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def _first_tensor_in(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = self._first_tensor_in(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for item in obj.values():
                t = self._first_tensor_in(item)
                if t is not None:
                    return t
        return None

    def _extract_input_tensors_for_layout(self, inputs) -> list:
        """Flatten inputs to a list of Tensors for MemoryLayoutAnalyzer."""
        tensors = []
        def walk(o):
            if isinstance(o, torch.Tensor):
                tensors.append(o)
            elif isinstance(o, (list, tuple)):
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                for x in o.values():
                    walk(x)
        walk(inputs)
        return tensors

    def _synthetic_report(self, cases: list, metadata: dict, op_type_info: dict) -> dict:
        """Used when no tensor outputs were analyzable (e.g. all outputs are
        scalars, or all paths had type/shape mismatches). Produces a minimal
        output_report entry that still satisfies Gate-F schema."""
        num_cases = metadata.get("num_cases", 0)
        return {
            "output_index": 0,
            "output_path": "output",
            "output_kind": "none",
            "output_shape": None,
            "output_dtype": None,
            "pass_fail": False,
            "basic_stats": {
                "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
                "median_abs_diff": 0.0, "p99_abs_diff": 0.0,
                "num_mismatched": 0, "total_elements": 0,
                "mismatch_ratio": 1.0, "match_rate": 0.0,
                "int8_special_tolerance": False,
            },
            "error_distribution": None,
            "value_range": None,
            "pattern_hint": {
                "primary_hint": "structural_mismatch",
                "primary_confidence": 0.5,
                "primary_evidence": "no analyzable tensor outputs found",
                "all_hints": [],
            },
            "worst_elements": [],
            "tail_analysis": {},
            "dimension_analysis": [],
            "L5_intermediate": None,
            "L7_code_mapping": None,
            "L8_op_type": op_type_info.get("op_type", "unknown"),
            "per_case": [],
            "representative_case_idx": 0,
            "case_aggregate": {
                "num_cases": num_cases,
                "mismatch_ratio_min": 1.0, "mismatch_ratio_max": 1.0,
                "mismatch_ratio_mean": 1.0,
                "max_abs_diff_min": 0.0, "max_abs_diff_max": 0.0,
                "pass_case_count": 0, "fail_case_count": num_cases,
                "all_cases_same_pattern": True,
                "shape_conditional": False,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="精度取证数值分析")
    parser.add_argument("task_name", help="task 目录名（相对于 repo root）")
    parser.add_argument(
        "--task-dir", default=None,
        help="task 绝对路径，默认为 {REPO_ROOT}/{task_name}",
    )
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    task_dir = args.task_dir or str(REPO_ROOT / args.task_name)
    PrecisionForensics(args.task_name, task_dir, args.attempt).run()


if __name__ == "__main__":
    main()
