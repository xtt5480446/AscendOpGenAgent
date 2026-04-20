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
            [sys.executable, str(VERIF_SCRIPT), self.op_name],
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
