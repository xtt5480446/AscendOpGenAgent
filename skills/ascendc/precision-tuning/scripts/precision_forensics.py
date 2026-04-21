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
SKILL_DIR    = SCRIPT_DIR.parent                        # precision-tuning/
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
                return {
                    "op_type": op_type,
                    "source": "name_heuristic",
                    "confidence": "low",
                    "pattern_priority": self.OP_TYPE_PATTERN_PRIORITY.get(op_type, []),
                }
        return {
            "op_type": "unknown",
            "source": "name_heuristic",
            "confidence": "low",
            "pattern_priority": [],
        }


# ============================================================
# 执行引擎 — 调用 verification_ascendc.py
# ============================================================

class OperatorExecutor:
    """调用 verification_ascendc.py 并解析 stdout 获取数值差异数据。"""

    def __init__(self, op_name: str, task_dir: str):
        self.op_name = op_name
        self.task_dir = Path(task_dir).resolve()

    def load_and_execute(self) -> dict:
        """
        运行 verification_ascendc.py，解析结构化 stdout。
        返回 {"stdout", "comparisons", "inputs_meta", "all_passed"}
        """
        result = subprocess.run(
            [sys.executable, str(VERIF_SCRIPT), str(self.task_dir)],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ,
                 "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")},
        )
        stdout = result.stdout
        comparisons = self._parse_comparisons(stdout)
        inputs_meta = self._parse_inputs(stdout)
        return {
            "stdout": stdout,
            "comparisons": comparisons,
            "inputs_meta": inputs_meta,
            "all_passed": result.returncode == 0,
        }

    def _parse_comparisons(self, stdout: str) -> list:
        """
        解析形如:
          case[0]: output[0]: dtype(ref=float16, cand=float16), unequal_elements=37,
                   mismatch_ratio=0.289062%, max_abs_diff=0.00390625, mean_abs_diff=0.00390625
        和:
          case[0]: output[0]: matched
        的行。
        """
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
                    "mismatch_ratio": float(m.group(5)),
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
                    results.append({
                        "name": m.group(1),
                        "shape": m.group(2),
                        "dtype": m.group(3),
                    })
        return results


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
        try:
            op_type_info = OperatorTypeDetector().detect(self.op_name, self.task_dir)
            data = OperatorExecutor(self.op_name, self.task_dir).load_and_execute()

            comparisons = data["comparisons"]
            by_output: dict = defaultdict(list)
            for c in comparisons:
                by_output[c["output_idx"]].append(c)

            output_reports = []
            for output_idx in sorted(by_output.keys()):
                comps = by_output[output_idx]
                n_fail = sum(1 for c in comps if not c["ok"])
                max_mismatch_pct = max(
                    (c.get("mismatch_ratio", 0) for c in comps), default=0.0
                )
                max_diff = max(
                    (c.get("max_abs_diff", 0.0) for c in comps), default=0.0
                )
                mean_diff = (
                    sum(c.get("mean_abs_diff", 0.0) for c in comps) / max(len(comps), 1)
                )
                mismatch_frac = max_mismatch_pct / 100.0
                match_rate = 1.0 - mismatch_frac

                pattern = self._classify_pattern(mismatch_frac, max_diff)
                basic_stats = {
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                    "median_abs_diff": None,
                    "p99_abs_diff": None,
                    "num_mismatched": n_fail,
                    "total_elements": len(comps),
                    "mismatch_ratio": mismatch_frac,
                    "match_rate": match_rate,
                }
                output_reports.append({
                    "pass_fail": n_fail == 0,
                    "basic_stats": basic_stats,
                    "error_distribution": None,
                    "value_range": None,
                    "pattern_hint": {
                        "primary_hint": pattern,
                        "primary_confidence": 0.5,
                        "primary_evidence": (
                            f"mismatch_ratio={max_mismatch_pct:.4f}%, "
                            f"max_abs_diff={max_diff}"
                        ),
                        "all_hints": [{
                            "pattern": pattern,
                            "confidence": 0.5,
                            "evidence": "parsed from verification_ascendc stdout",
                        }],
                    },
                    "worst_elements": [],
                    "tail_analysis": {},
                    "dimension_analysis": [],
                    "L5_intermediate": None,
                    "L7_code_mapping": None,
                    "L8_op_type": op_type_info.get("op_type", "unknown"),
                    "output_index": output_idx,
                    "output_shape": None,
                    "output_dtype": comps[0].get("dtype") if comps else None,
                })

            if not output_reports:
                mismatch_frac = 0.0 if data["all_passed"] else 1.0
                output_reports.append({
                    "pass_fail": data["all_passed"],
                    "basic_stats": {
                        "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
                        "median_abs_diff": None, "p99_abs_diff": None,
                        "num_mismatched": 0, "total_elements": 0,
                        "mismatch_ratio": mismatch_frac,
                        "match_rate": 1.0 - mismatch_frac,
                    },
                    "error_distribution": None, "value_range": None,
                    "pattern_hint": {
                        "primary_hint": "pass" if data["all_passed"] else "unknown",
                        "primary_confidence": 1.0 if data["all_passed"] else 0.1,
                        "primary_evidence": (
                            "all cases passed"
                            if data["all_passed"]
                            else "verification failed (no parsed output)"
                        ),
                        "all_hints": [],
                    },
                    "worst_elements": [], "tail_analysis": {}, "dimension_analysis": [],
                    "L5_intermediate": None, "L7_code_mapping": None,
                    "L8_op_type": op_type_info.get("op_type", "unknown"),
                    "output_index": 0, "output_shape": None, "output_dtype": None,
                })

            worst = max(output_reports, key=lambda r: r["basic_stats"]["mismatch_ratio"])
            ph = worst["pattern_hint"]
            comparator = HistoryComparator(self.tuning_dir, self.attempt)

            final = {
                "version": "2.0",
                "op_name": self.op_name,
                "attempt": self.attempt,
                "status": "completed",
                "num_outputs": len(output_reports),
                "L0_pass": data["all_passed"],
                "outputs": output_reports,
                "L5_intermediate": None,
                "L6_memory_layout": {
                    "inputs": data.get("inputs_meta", []),
                    "outputs": [],
                },
                "L7_code_mapping": None,
                "L8_operator": op_type_info,
                "primary_hint": ph["primary_hint"],
                "primary_confidence": ph["primary_confidence"],
                "primary_evidence": ph["primary_evidence"],
                "all_hints": ph["all_hints"],
                "history_trend": None,
                "multi_case_analysis": None,
                "num_test_cases": max(len(by_output), 1),
                "available_files": {
                    "reference": os.path.exists(os.path.join(self.task_dir, "model.py")),
                    "custom": os.path.exists(
                        os.path.join(self.task_dir, "model_new_ascendc.py")
                    ),
                },
            }

            trend = comparator.build_trend(final)
            if trend:
                final["history_trend"] = trend

            report_path = os.path.join(
                self.tuning_dir, f"forensics_report_{self.attempt}.json"
            )
            with open(report_path, "w") as f:
                json.dump(final, f, indent=2, ensure_ascii=False)

            stats = worst["basic_stats"]
            print(f"[FORENSICS] ✅ 精度取证完成 (attempt={self.attempt})")
            print(f"  op_type: {op_type_info['op_type']} (source={op_type_info['source']})")
            print(
                f"  primary_hint: {final['primary_hint']} "
                f"(confidence={final['primary_confidence']:.2f})"
            )
            print(f"  evidence: {final['primary_evidence']}")
            print(
                f"  mismatch: {stats['mismatch_ratio']:.2%} "
                f"({stats['num_mismatched']}/{stats['total_elements']})"
            )
            print(f"  max_diff: {stats['max_abs_diff']:.6f}")
            if trend:
                print(f"  趋势: {'↓ 改善中' if trend['mismatch_improving'] else '↑ 未改善'}")
            print(f"  report: {report_path}")
            return final

        except Exception as e:
            err = {
                "version": "2.0", "op_name": self.op_name, "attempt": self.attempt,
                "status": "error", "error": str(e), "traceback": traceback.format_exc(),
            }
            rp = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
            with open(rp, "w") as f:
                json.dump(err, f, indent=2, ensure_ascii=False)
            print(f"[FORENSICS] ❌ 取证失败: {e}", file=sys.stderr)
            sys.exit(1)

    def _classify_pattern(self, mismatch_frac: float, max_diff: float) -> str:
        if mismatch_frac >= 0.9:
            return "all_wrong"
        elif mismatch_frac >= 0.3:
            return "scattered"
        elif max_diff > 1.0:
            return "magnitude_correlated"
        elif mismatch_frac > 0:
            return "tail_spike"
        return "pass"


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
