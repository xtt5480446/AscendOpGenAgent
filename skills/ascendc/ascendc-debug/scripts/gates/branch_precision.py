"""branch_precision.py — 精度失败分支 Gate.

从原 precision_gate.py 的 `class GateChecker` **原样搬** precision 语义（findings.md §3.3 ③）:
  - check_forensics / _write_baseline_from_forensics
  - check_audit / 7 个 section 检查
  - check_validate + _compute_loop_signal + _count_stagnant + _detect_harmful_regression
    + _compute_improvement_ratio + _write_round_summary + _write_tuning_directions
    + _extract_direction_* + _check_direction_assessment + _extract_section
    + _extract_fix_type + _extract_changed_locations + _write_audit_index
    + _get_baseline_match_rate + _kernel_dir + _check_import_name_match + _result

对外契约:
  - `PrecisionBranch(op_name).run_gate_f(task_dir, attempt)`
  - `PrecisionBranch(op_name).run_gate_a(task_dir, attempt)`
  - `PrecisionBranch(op_name).run_gate_v(task_dir, attempt)`

所有文件产出路径、loop_signal 取值 (PASS/CONTINUE/STOP)、audit 7 section 名等均与搬移前一致。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .common import GateOutcome


MAX_ATTEMPTS = 2
MAX_STAGNANT_ROUNDS = 2


class _LegacyPrecisionChecker:
    """原 GateChecker 精度相关方法的 1:1 搬移。"""

    def __init__(self, op_name: str, task_dir: str, attempt: int = 0):
        self.op_name = op_name
        self.task_dir = task_dir
        self.attempt = attempt
        self.tuning_dir = os.path.join(task_dir, "precision_tuning")

    # ================================================================
    # Gate-F: 取证报告
    # ================================================================

    def check_forensics(self) -> dict:
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        checks = {
            "report_exists": os.path.exists(path),
            "report_parseable": False,
            "status_completed": False,
            "has_primary_hint": False,
            "has_outputs": False,
            "has_basic_stats": False,
            "attempt_matches": False,
        }
        r = None
        if checks["report_exists"]:
            try:
                with open(path) as f:
                    r = json.load(f)
                checks["report_parseable"] = True
                checks["status_completed"] = r.get("status") == "completed"
                checks["has_primary_hint"] = bool(r.get("primary_hint"))
                checks["has_outputs"] = len(r.get("outputs", [])) > 0
                if checks["has_outputs"]:
                    checks["has_basic_stats"] = "basic_stats" in r["outputs"][0]
                checks["attempt_matches"] = r.get("attempt", -1) == self.attempt
            except (json.JSONDecodeError, KeyError):
                r = None

        gate_result = self._result("GATE-F", checks)

        # Gate-F 通过且 attempt 0 时：从 forensics 写 baseline_state.json（幂等）
        if gate_result["passed"] and self.attempt == 0 and r is not None:
            self._write_baseline_from_forensics(r)

        return gate_result

    def _write_baseline_from_forensics(self, forensics: dict) -> None:
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            return

        try:
            outputs = forensics.get("outputs", [])
            if not outputs:
                return
            stats = outputs[0].get("basic_stats", {})
            raw_match_rate = stats.get("match_rate")
            raw_mismatch_ratio = stats.get("mismatch_ratio")
            if raw_match_rate is None:
                return

            baseline_match_rate = round(float(raw_match_rate) * 100, 4)
            baseline_mismatch_ratio = float(raw_mismatch_ratio) if raw_mismatch_ratio is not None else None

            baseline_state = {
                "match_rate":      baseline_match_rate,
                "mismatch_ratio":  baseline_mismatch_ratio,
                "max_abs_diff":    stats.get("max_abs_diff"),
                "mean_abs_diff":   stats.get("mean_abs_diff"),
                "primary_hint":    forensics.get("primary_hint"),
                "source":          "forensics_report.json/outputs[0]/basic_stats",
                "note":            "Initial precision captured at Gate-F before any code modification"
            }
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_state, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # ================================================================
    # Gate-A: 审计报告
    # ================================================================

    def check_audit(self) -> dict:
        prereq = self._check_prerequisite_forensics()
        if not prereq["satisfied"]:
            checks = {"prerequisite_forensics": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-A", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        checks = {
            "prerequisite_forensics": True,
            "report_exists": os.path.exists(path),
            "report_nonempty": False,
            "has_forensics_summary": False,
            "has_computation_decomposition": False,
            "has_reference_impl_spec": False,
            "has_kernel_step_trace": False,
            "has_root_cause": False,
            "has_fix_plan": False,
            "has_target_files": False,
            "has_direction_assessment": True,
        }
        content = None
        if checks["report_exists"]:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            checks["report_nonempty"] = len(content) > 200
            for tag, key in [("FORENSICS_SUMMARY", "has_forensics_summary"),
                             ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
                             ("REFERENCE_IMPL_SPEC", "has_reference_impl_spec"),
                             ("KERNEL_STEP_TRACE", "has_kernel_step_trace"),
                             ("ROOT_CAUSE", "has_root_cause"),
                             ("FIX_PLAN", "has_fix_plan"),
                             ("TARGET_FILES", "has_target_files")]:
                checks[key] = f"[{tag}]" in content
            checks["has_direction_assessment"] = (
                self.attempt == 0 or "[DIRECTION_ASSESSMENT]" in content
            )
            if self.attempt > 0:
                if "[DIRECTION_ASSESSMENT]" in content:
                    checks["direction_assessment_binary"] = self._validate_direction_binary(content)
                else:
                    checks["direction_assessment_binary"] = False

        gate_result = self._result("GATE-A", checks)

        if gate_result["passed"] and content:
            self._write_audit_index(content)

        return gate_result

    # ================================================================
    # Gate-V: 验证结果 + 循环控制
    # ================================================================

    def check_validate(self) -> dict:
        prereq = self._check_prerequisite_code()
        if not prereq["satisfied"]:
            checks = {"prerequisite_code": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-V", checks)
            result["prerequisite_error"] = prereq["reason"]
            result["loop_signal"] = "STOP"
            result["loop_reason"] = f"前置条件不满足: {prereq['reason']}"
            result["stop_reason_code"] = "prerequisite_failure"
            return result

        result_path = os.path.join(self.tuning_dir,
                                   f"validation_result_attempt_{self.attempt}.json")
        checks = {
            "prerequisite_code": True,
            "result_exists": os.path.exists(result_path),
            "result_parseable": False,
            "precision_passed": False,
        }

        correctness_passed = False
        if checks["result_exists"]:
            try:
                with open(result_path) as f:
                    r = json.load(f)
                checks["result_parseable"] = True
                correctness_passed = r.get("correctness_passed", False)
                checks["precision_passed"] = correctness_passed
            except (json.JSONDecodeError, KeyError):
                pass

        loop_signal, loop_reason, stop_reason_code = self._compute_loop_signal(correctness_passed)

        gate_result = self._result("GATE-V", checks)
        gate_result["loop_signal"] = loop_signal
        gate_result["loop_reason"] = loop_reason
        gate_result["stop_reason_code"] = stop_reason_code
        gate_result["attempt"] = self.attempt
        gate_result["max_attempts"] = MAX_ATTEMPTS

        self._write_round_summary(stop_reason_code)
        self._write_tuning_directions(stop_reason_code)

        return gate_result

    # ================================================================
    # 前置依赖检查
    # ================================================================

    def _check_prerequisite_forensics(self) -> dict:
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(path):
            return {"satisfied": False,
                    "reason": f"forensics_report_{self.attempt}.json 不存在, 必须先运行 precision_forensics.py",
                    "detail": {"forensics_exists": False, "forensics_attempt_match": False}}
        try:
            with open(path) as f:
                r = json.load(f)
            if r.get("status") != "completed":
                return {"satisfied": False,
                        "reason": f"forensics 状态异常: {r.get('status')}",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            if r.get("attempt", -1) != self.attempt:
                return {"satisfied": False,
                        "reason": f"forensics attempt={r.get('attempt')} 不匹配当前 attempt={self.attempt}, "
                                  f"必须重新运行 precision_forensics.py",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            return {"satisfied": True, "reason": "", "detail": {}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"forensics 解析失败: {e}",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": False}}

    def _check_prerequisite_code(self) -> dict:
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

    # ================================================================
    # 循环控制
    # ================================================================

    def _compute_loop_signal(self, passed: bool) -> tuple:
        if passed:
            return "PASS", "精度验证通过", "precision_passed"

        if self.attempt + 1 >= MAX_ATTEMPTS:
            return "STOP", f"已达最大轮次 ({MAX_ATTEMPTS})", "max_attempts_reached"

        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(forensics_path):
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                trend = fr.get("history_trend")
                if trend:
                    trend_list = trend.get("trend", [])

                    if self._detect_harmful_regression(trend_list):
                        return "STOP", "检测到 A→B→A 振荡型有害回退，需人工分析", "harmful_regression"

                    if not trend.get("mismatch_improving", True):
                        stagnant = self._count_stagnant(trend_list)
                        if stagnant >= MAX_STAGNANT_ROUNDS:
                            direction_ok = self._check_direction_assessment()
                            if direction_ok == "continue":
                                return "CONTINUE", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"但 Agent 已明确换方向，继续探索"
                                ), "stagnant_new_direction"
                            else:
                                return "STOP", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"Agent 仍沿用同一方向，可能方向错误，需人工分析"
                                ), "stagnant_same_direction"
            except (json.JSONDecodeError, KeyError):
                pass

        return "CONTINUE", f"精度未通过, 进入第 {self.attempt + 2} 轮", None

    def _count_stagnant(self, trend: list) -> int:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 2:
            return 0
        count = 0
        for i in range(len(ratios) - 1, 0, -1):
            if ratios[i] >= ratios[i - 1]:
                count += 1
            else:
                break
        return count

    def _detect_harmful_regression(self, trend: list) -> bool:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 3:
            return False
        r_prev, r_mid, r_curr = ratios[-3], ratios[-2], ratios[-1]
        mid_improved = (r_prev - r_mid) > 0.01
        curr_regressed = r_curr >= (r_prev - 0.005)
        return mid_improved and curr_regressed

    def _compute_improvement_ratio(self, prev_mismatch: float, curr_mismatch: float):
        prev_match = (1 - prev_mismatch) * 100
        curr_match = (1 - curr_mismatch) * 100
        remaining = 100 - prev_match
        if remaining <= 0:
            return None
        return round((curr_match - prev_match) / remaining, 4)

    def _write_round_summary(self, stop_reason_code) -> None:
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")

        if stop_reason_code is None:
            stop_reason_code = "validation_failed"

        existing = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        match_rate = None
        mismatch_ratio = None
        improvement_ratio = None
        absolute_improvement = None
        forensics_hint = None
        op_type = None

        result_path = os.path.join(self.tuning_dir, f"validation_result_attempt_{self.attempt}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    r = json.load(f)
                mr_str = r.get("match_rate")
                if mr_str is not None:
                    match_rate = round(float(mr_str), 4)
                    mismatch_ratio = round(1 - match_rate / 100, 8)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                pass

        if match_rate is not None:
            if self.attempt == 0:
                baseline_match_rate = self._get_baseline_match_rate()
                if baseline_match_rate is not None:
                    baseline_mismatch = 1 - baseline_match_rate / 100
                    curr_mismatch = 1 - match_rate / 100
                    improvement_ratio = self._compute_improvement_ratio(
                        baseline_mismatch, curr_mismatch
                    )
                    absolute_improvement = round(match_rate - baseline_match_rate, 4)
            elif self.attempt > 0:
                prev_result_path = os.path.join(
                    self.tuning_dir, f"validation_result_attempt_{self.attempt - 1}.json"
                )
                if os.path.exists(prev_result_path):
                    try:
                        with open(prev_result_path) as f:
                            prev_r = json.load(f)
                        prev_mr_str = prev_r.get("match_rate")
                        if prev_mr_str is not None:
                            prev_match_rate = float(prev_mr_str)
                            prev_mismatch = 1 - prev_match_rate / 100
                            curr_mismatch = 1 - match_rate / 100
                            improvement_ratio = self._compute_improvement_ratio(
                                prev_mismatch, curr_mismatch
                            )
                            absolute_improvement = round(match_rate - prev_match_rate, 4)
                    except (json.JSONDecodeError, KeyError, OSError, ValueError):
                        pass

        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(forensics_path):
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                forensics_hint = fr.get("primary_hint")
                op_type = fr.get("op_type") or fr.get("L8_operator", {}).get("op_type")
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        compile_log_abs = os.path.join(
            self.tuning_dir, f"compilation_log_{self.attempt}.json"
        )
        compilation_log_ref = (
            f"precision_tuning/compilation_log_{self.attempt}.json"
            if os.path.exists(compile_log_abs) else None
        )

        summary = dict(existing)
        summary["attempt"] = self.attempt

        metrics = summary.get("metrics", {})
        metrics.update({
            "match_rate":            match_rate,
            "mismatch_ratio":        mismatch_ratio,
            "improvement_ratio":     improvement_ratio,
            "absolute_improvement":  absolute_improvement,
            "stop_reason_code":      stop_reason_code,
        })
        summary["metrics"] = metrics

        diagnostics = summary.get("diagnostics", {})
        diagnostics["forensics_hint"] = forensics_hint
        diagnostics["op_type"] = op_type
        summary["diagnostics"] = diagnostics

        index = summary.get("index", {})
        index["compilation_log"] = compilation_log_ref
        summary["index"] = index

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _write_tuning_directions(self, stop_reason_code) -> None:
        directions_path = os.path.join(self.tuning_dir, "tuning_directions.json")

        data = {"op_name": self.op_name, "final_status": "in_progress", "entries": []}
        if os.path.exists(directions_path):
            try:
                with open(directions_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        fix_type = None
        direction_verdict = None
        forensics_hint = None
        improvement_ratio = None
        absolute_improvement = None
        match_rate = None
        mismatch_ratio = None

        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
                diag = summary.get("diagnostics", {})
                fix_type = diag.get("fix_type")
                direction_verdict = diag.get("direction_verdict")
                forensics_hint = diag.get("forensics_hint")
                metrics = summary.get("metrics", {})
                improvement_ratio = metrics.get("improvement_ratio")
                absolute_improvement = metrics.get("absolute_improvement")
                match_rate = metrics.get("match_rate")
                mismatch_ratio = metrics.get("mismatch_ratio")
            except (json.JSONDecodeError, OSError):
                pass

        if stop_reason_code == "precision_passed":
            outcome = "passed"
        elif improvement_ratio is None:
            outcome = "stagnant"
        elif improvement_ratio < -0.05:
            outcome = "regressed"
        elif improvement_ratio >= 0.1:
            outcome = "improved"
        else:
            outcome = "stagnant"

        direction_reason = self._extract_direction_reason()

        new_entry = {
            "attempt":               self.attempt,
            "fix_type":              fix_type,
            "forensics_hint":        forensics_hint,
            "direction_verdict":     direction_verdict,
            "direction_reason":      direction_reason,
            "improvement_ratio":     improvement_ratio,
            "absolute_improvement":  absolute_improvement,
            "outcome":               outcome,
            "evidence": {
                "forensics_ref":     f"precision_tuning/forensics_report_{self.attempt}.json",
                "audit_ref":         f"precision_tuning/precision_audit_{self.attempt}.md",
                "match_rate":        match_rate,
                "mismatch_ratio":    mismatch_ratio,
            }
        }

        data["entries"] = [e for e in data["entries"] if e.get("attempt") != self.attempt]
        data["entries"].append(new_entry)
        data["entries"].sort(key=lambda e: e.get("attempt", 0))

        terminal_codes = {
            "max_attempts_reached", "stagnant_same_direction",
            "stagnant_new_direction", "harmful_regression", "prerequisite_failure",
        }
        if stop_reason_code == "precision_passed":
            data["final_status"] = "success"
            for entry in data["entries"]:
                ir = entry.get("improvement_ratio")
                same_fix = entry.get("fix_type") == fix_type
                nonneg = ir is None or ir >= 0
                entry["contributed"] = same_fix and nonneg
            for entry in data["entries"]:
                if entry.get("attempt") == self.attempt:
                    entry["contributed"] = True
        elif stop_reason_code in terminal_codes:
            data["final_status"] = "failed"

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(directions_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _extract_direction_reason(self):
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return None

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return None
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()

            if not section:
                return None

            for line in section.split("\n"):
                if "换方向理由" in line:
                    colon_pos = line.find(":")
                    if colon_pos == -1:
                        continue
                    reason = line[colon_pos + 1:].strip()
                    return reason if reason else None

            return None
        except (OSError, UnicodeDecodeError):
            return None

    def _check_direction_assessment(self) -> str:
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return "unknown"

        try:
            with open(path) as f:
                content = f.read()

            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return "unknown"
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()

            if not section:
                return "unknown"

            for line in section.split("\n"):
                key = "本轮是否延续上一轮方向"
                if key not in line and "本轮是否延续" not in line:
                    continue
                colon_pos = line.find(":")
                if colon_pos == -1:
                    continue
                answer = line[colon_pos + 1:].strip()
                first_word = self._extract_direction_first_word(answer)
                if first_word == "否":
                    return "continue"
                elif first_word == "是":
                    return "stop"
            return "unknown"
        except (OSError, UnicodeDecodeError):
            return "unknown"

    def _extract_direction_first_word(self, text: str) -> str:
        if not text:
            return ""
        first = text.split()[0] if text.split() else text
        return first.rstrip("，。！？、；：…,.")

    def _validate_direction_binary(self, content: str) -> bool:
        marker = "[DIRECTION_ASSESSMENT]"
        start = content.find(marker)
        if start == -1:
            return False
        start += len(marker)
        next_bracket = content.find("\n[", start)
        section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            return first_word in ("是", "否")
        return False

    # ================================================================
    # Section 提取与 audit index 写入
    # ================================================================

    def _extract_section(self, content: str, section_name: str):
        marker = f"[{section_name}]"
        start = content.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end_marker = content.find("\n[", start)
        end_audit = content.find("=== END AUDIT ===", start)
        candidates = [pos for pos in [end_marker, end_audit] if pos != -1]
        end = min(candidates) if candidates else len(content)
        text = content[start:end].strip()
        return text if text else None

    def _extract_fix_type(self, content: str):
        section = self._extract_section(content, "FIX_PLAN")
        if not section:
            return None
        m = re.search(r"FIX_PRECISION_\w+", section)
        return m.group(0) if m else None

    def _extract_changed_locations(self, content: str) -> list:
        section = self._extract_section(content, "TARGET_FILES")
        if not section:
            return []
        locations = []
        seen = set()
        for line in section.split("\n"):
            line = line.strip().lstrip("-*•·").strip()
            for part in line.split():
                part = part.rstrip(",:;")
                if re.search(r"\.\w{1,5}$", part) and part not in seen:
                    locations.append(part)
                    seen.add(part)
        return locations

    def _extract_direction_verdict_value(self, content: str):
        if self.attempt == 0:
            return None
        section = self._extract_section(content, "DIRECTION_ASSESSMENT")
        if not section:
            return None
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            if first_word in ("是", "否"):
                return first_word
        return None

    def _write_audit_index(self, content: str) -> None:
        attempt_dir = os.path.join(self.tuning_dir, "history", f"attempt_{self.attempt}")
        sections_dir = os.path.join(attempt_dir, "sections")
        try:
            os.makedirs(sections_dir, exist_ok=True)
        except OSError:
            return

        SECTION_MAP = [
            ("forensics_summary",        "FORENSICS_SUMMARY"),
            ("computation_decomposition","COMPUTATION_DECOMPOSITION"),
            ("reference_impl_spec",      "REFERENCE_IMPL_SPEC"),
            ("kernel_step_trace",        "KERNEL_STEP_TRACE"),
            ("knowledge_match",          "KNOWLEDGE_MATCH"),
            ("root_cause",               "ROOT_CAUSE"),
            ("fix_plan",                 "FIX_PLAN"),
            ("target_files",             "TARGET_FILES"),
            ("direction_assessment",     "DIRECTION_ASSESSMENT"),
        ]

        sections_index = {}
        base = f"precision_tuning/history/attempt_{self.attempt}/sections"
        for key, tag in SECTION_MAP:
            sec_text = self._extract_section(content, tag)
            rel_path = f"{base}/{key}.md"
            if sec_text is not None:
                abs_path = os.path.join(sections_dir, f"{key}.md")
                try:
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(f"[{tag}]\n\n{sec_text}\n")
                    sections_index[key] = rel_path
                except OSError:
                    sections_index[key] = None
            else:
                sections_index[key] = None

        diagnostics = {
            "forensics_hint":    None,
            "op_type":           None,
            "fix_type":          self._extract_fix_type(content),
            "changed_locations": self._extract_changed_locations(content),
            "direction_verdict": self._extract_direction_verdict_value(content),
        }

        n = self.attempt
        index = {
            "forensics":          f"precision_tuning/history/attempt_{n}/forensics_report.json",
            "audit_full":         f"precision_tuning/precision_audit_{n}.md",
            "sections":           sections_index,
            "code_snapshot":      f"precision_tuning/history/attempt_{n}/code_snapshot/",
            "validation":         f"precision_tuning/validation_result_attempt_{n}.json",
            "compilation_log":    None,
            "tuning_directions":  "precision_tuning/tuning_directions.json",
            "forensics_used":     f"precision_tuning/forensics_report_{n}.json",
        }

        initial_summary = {
            "attempt": self.attempt,
            "metrics": {
                "match_rate":            None,
                "mismatch_ratio":        None,
                "improvement_ratio":     None,
                "absolute_improvement":  None,
                "stop_reason_code":      None,
            },
            "diagnostics": diagnostics,
            "index": index,
        }
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(initial_summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # ================================================================
    # 工具
    # ================================================================

    def _get_baseline_match_rate(self):
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path) as f:
                    bs = json.load(f)
                mr = bs.get("match_rate")
                if mr is not None:
                    return float(mr)
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(forensics_path):
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)

                history_trend = fr.get("history_trend")
                if history_trend:
                    trend_list = history_trend.get("trend", [])
                    if len(trend_list) >= 2:
                        baseline_mismatch = trend_list[0].get("mismatch_ratio")
                        if baseline_mismatch is not None:
                            return round((1 - float(baseline_mismatch)) * 100, 4)

                if self.attempt == 0:
                    outputs = fr.get("outputs", [])
                    if outputs:
                        raw_mr = outputs[0].get("basic_stats", {}).get("match_rate")
                        if raw_mr is not None:
                            return round(float(raw_mr) * 100, 4)

            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass

        return None

    def _kernel_dir(self):
        kdir = os.path.join(self.task_dir, "kernel")
        return kdir if os.path.isdir(kdir) else None

    def _check_import_name_match(self) -> bool:
        wrapper = os.path.join(self.task_dir, "model_new_ascendc.py")
        pybind  = os.path.join(self.task_dir, "kernel", "pybind11.cpp")
        if not os.path.exists(wrapper) or not os.path.exists(pybind):
            return False
        wrapper_text = open(wrapper).read()
        import_m = re.search(r"import\s+(_\w+)", wrapper_text)
        if not import_m:
            return False
        import_name = import_m.group(1)
        pybind_text = open(pybind).read()
        module_m = re.search(r"PYBIND11_MODULE\s*\(\s*(\w+)\s*,", pybind_text)
        if not module_m:
            return False
        module_name = "_" + module_m.group(1)
        return import_name == module_name

    def _result(self, gate_name: str, checks: dict) -> dict:
        return {"gate": gate_name, "passed": all(checks.values()), "checks": checks}


def _legacy_to_outcome(raw: dict) -> GateOutcome:
    """把 _LegacyPrecisionChecker 返回的 dict (gate/passed/checks[/loop_*]) 转成 GateOutcome。"""
    checks = dict(raw.get("checks", {}))
    # 附带 prerequisite_error / stop_reason_code / attempt / max_attempts 等保留到 checks
    for k in ("prerequisite_error", "stop_reason_code", "attempt", "max_attempts"):
        if k in raw:
            checks[k] = raw[k]
    return GateOutcome(
        gate=raw.get("gate", ""),
        ok=bool(raw.get("passed", False)),
        checks=checks,
        loop_signal=raw.get("loop_signal"),
        reason=raw.get("loop_reason"),
    )


class PrecisionBranch:
    """精度分支的 Gate 入口。与其它 branch_* 的签名对齐。

    方法签名: run_gate_f(task_dir, attempt) / run_gate_a(...) / run_gate_v(...)
    task_dir: pathlib.Path 或 str 都接受
    """

    def __init__(self, op_name: str):
        self.op_name = op_name

    def _checker(self, task_dir, attempt: int) -> _LegacyPrecisionChecker:
        return _LegacyPrecisionChecker(self.op_name, str(task_dir), attempt)

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        raw = self._checker(task_dir, attempt).check_forensics()
        return _legacy_to_outcome(raw)

    def run_gate_a(self, task_dir, attempt: int) -> GateOutcome:
        raw = self._checker(task_dir, attempt).check_audit()
        return _legacy_to_outcome(raw)

    def run_gate_v(self, task_dir, attempt: int) -> GateOutcome:
        raw = self._checker(task_dir, attempt).check_validate()
        return _legacy_to_outcome(raw)
