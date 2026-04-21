# Precision Forensics Rewrite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore full L0-L8 precision forensics analysis (DiffAnalyzer / MemoryLayoutAnalyzer / OutputFlattener) to `precision_forensics.py` while keeping JSON schema backward compatible with `precision_gate.py` and execution-semantic parity with `utils/verification_ascendc.py`.

**Architecture:** 3-process model (parent analyzer / child executor via subprocess / parity test).  Child executor copies loader helpers from `verification_ascendc.py`, runs model via spawned python subprocess, pickle-dumps tensors; parent loads + flattens outputs + runs DiffAnalyzer per canonical path.  All changes strictly additive on top of existing JSON schema.

**Tech Stack:** Python 3.11+, torch/torch_npu, numpy, pickle, pytest, subprocess.

---

## Phase 1 — Pure numpy analyzers (zero NPU, runnable locally)

### Task 1.1: Add DiffAnalyzer

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py` — append new class

**Step 1: Port DiffAnalyzer from original repo**

Copy `class DiffAnalyzer` (and its helpers `_basic_stats`, `_error_distribution`, `_value_range`, `_classify_pattern`, `_check_tail_spike`, `_check_dim_concentration`, `_check_magnitude_correlation`, `_check_boundary_concentration`, `_worst_elements`, `_tail_analysis`, `_dimension_analysis`) from `/Users/junming/code/operator/OpenOps/OpenOps-debug/.opencode/skills/precision-tuning/scripts/precision_forensics.py:346-620` **verbatim** into the new file, with additions:
- `basic_stats` dict gets new key `"int8_special_tolerance": False` (filled by caller)
- `value_range` dict gets NaN/Inf counts pre-nan-to-num
- Add module docstring explaining ATOL/RTOL must mirror `verification_ascendc.py`

Do not call it from anywhere yet (dead code for now).

**Step 2: Sanity test**

```bash
cd /Users/junming/code/operator/AscendOpGenAgent
python3 -c "
import numpy as np
import sys; sys.path.insert(0, 'skills/ascendc/precision-tuning/scripts')
from precision_forensics import DiffAnalyzer
g = np.random.randn(4, 8).astype(np.float32)
a = g + 0.005
r = DiffAnalyzer().analyze(g, a)
assert 'sign_analysis' in r['error_distribution']
assert len(r['worst_elements']) == 10
assert 'tile_8' in r['tail_analysis'] or r['tail_analysis'].get('note')
print('PASS')
"
```

Expected: `PASS`

**Step 3: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "forensics: add DiffAnalyzer class (dead code, Phase 1/3)"
```

---

### Task 1.2: Add MemoryLayoutAnalyzer

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py` — append new class

**Step 1: Port MemoryLayoutAnalyzer**

Copy `class MemoryLayoutAnalyzer` from `/Users/junming/code/operator/OpenOps/OpenOps-debug/.opencode/skills/precision-tuning/scripts/precision_forensics.py:626-650` **verbatim**.  No behavioral change.

**Step 2: Sanity test**

```bash
cd /Users/junming/code/operator/AscendOpGenAgent
python3 -c "
import torch, sys
sys.path.insert(0, 'skills/ascendc/precision-tuning/scripts')
from precision_forensics import MemoryLayoutAnalyzer
t = torch.randn(2, 3, 64)
r = MemoryLayoutAnalyzer().analyze_tensors([t], 'test')
assert r[0]['shape'] == [2, 3, 64]
assert r[0]['last_dim_alignment']['tile_32']['aligned'] == True
print('PASS')
"
```

Expected: `PASS`

**Step 3: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "forensics: add MemoryLayoutAnalyzer class (dead code, Phase 1/3)"
```

---

### Task 1.3: Add OutputFlattener

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py` — append new class

**Step 1: Implement OutputFlattener.flatten(ref, cand, root="output")**

Handle recursive `Tensor | list | tuple | dict | scalar`.  Path syntax aligned with `verification_ascendc._compare_values` (lines 146-185 of `utils/verification_ascendc.py`):
- `output[0]` — top-level tensor
- `output[1].foo` — dict key
- `output[2][3]` — list/tuple index

Output: `dict[path_str, {"ref": Tensor|scalar|None, "cand": Tensor|scalar|None, "kind": str, "shape": list|None, "dtype": str|None, "status": "ok"|"type_mismatch"|"shape_mismatch"|"len_mismatch"|"key_mismatch"}]`

Kind values: `"tensor" | "scalar" | "none"`.

**Step 2: Test**

```bash
cd /Users/junming/code/operator/AscendOpGenAgent
python3 -c "
import torch, sys
sys.path.insert(0, 'skills/ascendc/precision-tuning/scripts')
from precision_forensics import OutputFlattener

# Case A: single tensor
ref = torch.zeros(3, 4)
cand = torch.zeros(3, 4)
f = OutputFlattener().flatten(ref, cand)
assert 'output' in f, f
assert f['output']['kind'] == 'tensor'

# Case B: tuple of tensors
ref = (torch.zeros(2), torch.zeros(3, 4))
cand = (torch.zeros(2), torch.zeros(3, 4))
f = OutputFlattener().flatten(ref, cand)
assert 'output[0]' in f and 'output[1]' in f, f

# Case C: dict
ref = {'a': torch.zeros(2), 'b': torch.zeros(3)}
cand = {'a': torch.zeros(2), 'b': torch.zeros(3)}
f = OutputFlattener().flatten(ref, cand)
assert 'output.a' in f, f

# Case D: nested (list of dicts)
ref = [{'x': torch.zeros(2)}, {'y': torch.zeros(3)}]
cand = [{'x': torch.zeros(2)}, {'y': torch.zeros(3)}]
f = OutputFlattener().flatten(ref, cand)
assert 'output[0].x' in f and 'output[1].y' in f, f

# Case E: shape mismatch
ref = torch.zeros(3, 4)
cand = torch.zeros(3, 5)
f = OutputFlattener().flatten(ref, cand)
assert f['output']['status'] == 'shape_mismatch', f

# Case F: type mismatch
ref = torch.zeros(3)
cand = [torch.zeros(3)]
f = OutputFlattener().flatten(ref, cand)
assert 'output' in f and f['output']['status'] == 'type_mismatch', f

print('PASS')
"
```

Expected: `PASS`

**Step 3: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "forensics: add OutputFlattener for canonical-path nested-output support"
```

---

## Phase 2 — Child executor + parity scaffolding

### Task 2.1: Create `_forensics_child.py` with copied loader

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/_forensics_child.py`

**Step 1: Copy loader helpers verbatim**

From `utils/verification_ascendc.py` copy these functions into `_forensics_child.py` with a clear boundary comment block (`# ── BEGIN COPIED FROM utils/verification_ascendc.py ──` / `# ── END ──`):
- `_load_module` (lines 17-25)
- `_find_model_class` (lines 28-37)
- `_clone_value` (lines 40-49)
- `_move_to_device` (lines 52-61)
- `_normalize_output` (lines 64-73)
- `_contains_int8_tensor` (lines 76-86)
- `_get_input_groups` (lines 233-247)
- `_get_device` (lines 188-194)

**Step 2: Implement main `run()`**

Signature: `def run(task_dir: str, dump_dir: str) -> int`

Behavior:
1. `ref_module = _load_module(task_dir/model.py, ...)`, `cand_module = _load_module(task_dir/model_new_ascendc.py, ...)`
2. Inject `task_dir`, `task_dir/kernel/build` into `sys.path`
3. Set `torch.manual_seed(0)`
4. Get `init_inputs` (cand-priority; see verification_ascendc.py lines 309-312)
5. `input_groups = _get_input_groups(ref_module)`
6. `device = _get_device()`
7. Instantiate `ref_model = ref_cls(*init).to(device).eval()`, `cand_model = cand_cls(*init).to(device).eval()`
8. `os.makedirs(dump_dir, exist_ok=True)`
9. For each `case_idx, inputs in enumerate(input_groups)`:
    - Clone + move to device
    - `ref_out = ref_model(*ref_in)`, `cand_out = cand_model(*cand_in)` under `torch.no_grad`
    - `torch_npu.npu.synchronize()` if device is npu
    - `ref_out_cpu = _normalize_output(ref_out)`, `cand_out_cpu = _normalize_output(cand_out)`
    - `int8_triggered = _contains_int8_tensor(ref_out_cpu) and _contains_int8_tensor(cand_out_cpu)`
    - pickle dump `{ref, cand, inputs, int8_triggered, atol, rtol}` to `{dump_dir}/case_{case_idx}.pkl`
    - `del ref_out, cand_out, ref_in, cand_in; torch_npu.npu.empty_cache() if applicable`
10. Write `{dump_dir}/metadata.json` with `{num_cases, device, atol_default, rtol_default, parity_hash}`
11. Return 0

On any exception: print stacktrace to stderr, return 1.

**Step 3: Sanity test syntax**

```bash
python3 -c "import ast; ast.parse(open('skills/ascendc/precision-tuning/scripts/_forensics_child.py').read()); print('PARSE OK')"
```

**Step 4: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/_forensics_child.py
git commit -m "forensics: add _forensics_child.py executor (copied loader from verification_ascendc)"
```

---

### Task 2.2: Enrich OperatorTypeDetector with attributes + reduction_axis

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py` — `class OperatorTypeDetector`

**Step 1: Add `_extract_attributes(task_dir)` method**

Implementation: AST-parse `{task_dir}/model.py`, find `class Model(nn.Module): def __init__(self, ...)`, extract default argument values for common attrs:
- `kernel_size`, `stride`, `padding`, `dilation`, `groups`
- `dim`, `axis`, `keepdim`, `reduce`
- `normalized_shape`, `eps`, `elementwise_affine`
- `alpha`, `beta`, `gamma`
- `num_features`, `num_classes`, `in_features`, `out_features`

Return dict of found attrs. Failure → empty dict (not fatal).

**Step 2: Add `_infer_reduction_axis(task_dir, attrs)`**

Only called for op_type in `{"reduction", "normalization", "pooling"}`.  Try to infer:
- From `attrs["dim"]` or `attrs["axis"]` → axis index
- From first input_group's tensor shape → axis_length
- Return `{"axis_index": int, "axis_length": int}` or None

If input_groups unavailable statically, set `axis_length` to None (Sub-step 2.1 fallback).

**Step 3: Update `detect()` to call new methods**

Add `attributes` and `reduction_axis` keys to return dict.

**Step 4: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "forensics: enrich OperatorTypeDetector with attributes + reduction_axis"
```

---

### Task 2.3: Create parity test

**Files:**
- Create: `skills/ascendc/precision-tuning/scripts/test_executor_parity.py`

**Step 1: Test scaffold**

Parity test spec:
- Fixture list: `["avg_pool3_d", "rms_norm"]` under `archive_tasks/`; add synthetic cases for nested output, int8
- For each fixture:
    1. Run `utils/verification_ascendc.py <task>` via subprocess, parse stdout for `case[i]: ...mismatch_ratio=X%, max_abs_diff=Y`
    2. Run `_forensics_child.py <task> <tmpdir>` via subprocess, load pickled tensors, compute `mismatch_ratio` using **same** torch.nan_to_num + atol/rtol logic as verification (`_tensor_diff_summary`)
    3. Assert pass/fail matches
    4. Assert per-case `mismatch_ratio` matches within 1e-6 tolerance
    5. Assert `OutputFlattener(ref, cand).keys()` produces canonical path set matching `_compare_values` output
- Skip fixture if `kernel/build` missing (print SKIP)

Minimal version: just wire the subprocess calls, parse stdout, assert pass/fail. Advanced per-case equality left for follow-up.

**Step 2: Does not run locally** (needs NPU). Document in file header: `# NOTE: run inside cjm_cann* container via ssh npu_server "docker exec ... python3 test_executor_parity.py"`.

**Step 3: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/test_executor_parity.py
git commit -m "forensics: add parity test scaffold (NPU-dependent)"
```

---

## Phase 3 — Wire PrecisionForensics.run to new pipeline

### Task 3.1: Rewrite `OperatorExecutor` to use subprocess

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py` — `class OperatorExecutor`

**Step 1: New `load_and_execute()` impl**

```python
def load_and_execute(self) -> dict:
    dump_dir = Path(self.task_dir) / "precision_tuning" / f".forensics_tmp_{self.attempt}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    child = Path(__file__).parent / "_forensics_child.py"
    proc = subprocess.run(
        [sys.executable, str(child), self.task_dir, str(dump_dir)],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"forensics child failed (rc={proc.returncode}): {proc.stderr[-2000:]}")

    metadata = json.loads((dump_dir / "metadata.json").read_text())
    cases = []
    for case_idx in range(metadata["num_cases"]):
        case_path = dump_dir / f"case_{case_idx}.pkl"
        with case_path.open("rb") as f:
            cases.append(pickle.load(f))
    return {"cases": cases, "metadata": metadata}
```

**Step 2: Remove stdout-parsing logic (`_parse_comparisons`, `_parse_inputs`)**

These are no longer needed; delete.

**Step 3: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "forensics: switch OperatorExecutor to subprocess-based tensor dump"
```

---

### Task 3.2: Rewrite `PrecisionForensics.run` to assemble new JSON

**Files:**
- Modify: `skills/ascendc/precision-tuning/scripts/precision_forensics.py` — `class PrecisionForensics.run`

**Step 1: New assembly logic**

```python
def run(self) -> dict:
    try:
        op_type_info = OperatorTypeDetector().detect(self.op_name, self.task_dir)
        data = OperatorExecutor(self.op_name, self.task_dir, self.attempt).load_and_execute()
        flattener = OutputFlattener()
        analyzer = DiffAnalyzer(op_type_info=op_type_info)
        layout = MemoryLayoutAnalyzer()

        # Step 1: flatten each case's outputs → set of paths (stable across cases)
        per_case_paths = []
        for case in data["cases"]:
            flat = flattener.flatten(case["ref"], case["cand"], root="output")
            per_case_paths.append(flat)

        # Step 2: union of paths (some cases may have missing keys due to dynamic output structure)
        all_paths = sorted(set().union(*[set(p.keys()) for p in per_case_paths]))

        # Step 3: per-path, per-case diff
        output_reports = []
        int8_any_active = False
        nan_inf_agg = {"ref": {"has_nan": False, "has_inf": False, "nan_count": 0, "inf_count": 0},
                       "cand": {"has_nan": False, "has_inf": False, "nan_count": 0, "inf_count": 0}}

        for output_index, path in enumerate(all_paths):
            per_case_diffs = []
            for case_idx, case in enumerate(data["cases"]):
                if path not in per_case_paths[case_idx]:
                    continue
                pcp = per_case_paths[case_idx][path]
                if pcp["kind"] != "tensor" or pcp["status"] != "ok":
                    # skip; record as degenerate per-case entry
                    continue
                ref_np = pcp["ref"].float().numpy()
                cand_np = pcp["cand"].float().numpy()
                # comparison semantics: follow verification_ascendc (nan_to_num before mask)
                ref_safe = np.nan_to_num(ref_np, nan=0.0, posinf=1e9, neginf=-1e9)
                cand_safe = np.nan_to_num(cand_np, nan=0.0, posinf=1e9, neginf=-1e9)
                atol = 1.5 if case["int8_triggered"] else 1e-2
                rtol = 0.0 if case["int8_triggered"] else 1e-2
                analyzer.ATOL = atol; analyzer.RTOL = rtol
                diff = analyzer.analyze(ref_safe, cand_safe)
                diff["basic_stats"]["int8_special_tolerance"] = case["int8_triggered"]
                diff["case_idx"] = case_idx
                per_case_diffs.append(diff)
                if case["int8_triggered"]:
                    int8_any_active = True
                # track NaN/Inf from raw (pre-nan-to-num) tensors
                # accumulate into nan_inf_agg

            if not per_case_diffs:
                continue
            # representative = max mismatch_ratio case
            rep_idx = max(range(len(per_case_diffs)),
                          key=lambda i: (per_case_diffs[i]["basic_stats"]["mismatch_ratio"],
                                         per_case_diffs[i]["basic_stats"]["max_abs_diff"]))
            rep = per_case_diffs[rep_idx]
            agg = self._compute_case_aggregate(per_case_diffs, data["cases"])
            first_pcp = next(iter(p for p in per_case_paths if path in p), {}).get(path, {})
            output_reports.append({
                "output_index": output_index,
                "output_path": path,
                "output_kind": first_pcp.get("kind", "unknown"),
                "output_shape": first_pcp.get("shape"),
                "output_dtype": first_pcp.get("dtype"),
                "pass_fail": all(d["basic_stats"]["num_mismatched"] == 0 for d in per_case_diffs),
                **{k: rep[k] for k in
                   ("basic_stats", "error_distribution", "value_range", "pattern_hint",
                    "worst_elements", "tail_analysis", "dimension_analysis",
                    "L5_intermediate", "L7_code_mapping", "L8_op_type")},
                "per_case": per_case_diffs,
                "representative_case_idx": per_case_diffs[rep_idx]["case_idx"],
                "case_aggregate": agg,
            })

        # Top-level fields
        inputs_layout = layout.analyze_tensors(data["cases"][0]["inputs"], "input") if data["cases"] else []

        if output_reports:
            worst = max(output_reports, key=lambda o: (
                o["case_aggregate"]["mismatch_ratio_max"],
                o["case_aggregate"]["max_abs_diff_max"]))
            ph = worst["pattern_hint"]
        else:
            ph = {"primary_hint": "pass", "primary_confidence": 1.0,
                  "primary_evidence": "all outputs passed", "all_hints": []}

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
                "inputs": inputs_layout,
                "outputs": [],  # per-output layout lives in outputs[i].output_shape
            },
            "L7_code_mapping": None,
            "L8_operator": op_type_info,
            "primary_hint": ph["primary_hint"],
            "primary_confidence": ph["primary_confidence"],
            "primary_evidence": ph["primary_evidence"],
            "all_hints": ph.get("all_hints", []),
            "history_trend": None,
            "multi_case_analysis": None,
            "num_test_cases": data["metadata"]["num_cases"],
            "available_files": {
                "reference": os.path.exists(os.path.join(self.task_dir, "model.py")),
                "custom":    os.path.exists(os.path.join(self.task_dir, "model_new_ascendc.py")),
            },
            "int8_path_active": int8_any_active,
            "nan_inf_detected": nan_inf_agg,
        }

        # History trend
        trend = HistoryComparator(self.tuning_dir, self.attempt).build_trend(final)
        if trend:
            final["history_trend"] = trend

        # Write report
        report_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        with open(report_path, "w") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

        # Print summary (unchanged format)
        ...

        return final

    except Exception as e:
        err = {"version": "2.0", "op_name": self.op_name, "attempt": self.attempt,
               "status": "error", "error": str(e), "traceback": traceback.format_exc(),
               "outputs": [], "primary_hint": "error"}
        report_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        with open(report_path, "w") as f:
            json.dump(err, f, indent=2, ensure_ascii=False)
        return err
```

Also implement helper `_compute_case_aggregate(per_case_diffs, cases) -> dict`.

**Step 2: Commit**

```bash
git add skills/ascendc/precision-tuning/scripts/precision_forensics.py
git commit -m "forensics: rewrite PrecisionForensics.run with DiffAnalyzer + case aggregate"
```

---

### Task 3.3: Remove stale decomposition (`_classify_pattern` at module top)

The current simplified `PrecisionForensics._classify_pattern(self, mismatch_frac, max_diff)` (lines 436-446 of current file) is obsolete once DiffAnalyzer does classification. Delete.

Also delete stale stdout-parsing helpers if any remain.

---

## Phase 4 — Remote validation (NPU)

### Task 4.1: Push + pull on remote

```bash
git push
ssh npu_server "cd /home/c00959374/AscendOpGenAgent && git pull origin cjm/debug"
```

### Task 4.2: Run parity test inside container

```bash
ssh npu_server 'docker exec cjm_cann1 bash -c "
  cd /home/c00959374/AscendOpGenAgent
  python3 skills/ascendc/precision-tuning/scripts/test_executor_parity.py
"'
```

Expected: all fixtures PASS (or SKIP if `kernel/build` missing — acceptable for archive fixtures).

### Task 4.3: Run full forensics on 3_Add_pt1

```bash
ssh npu_server 'docker exec cjm_cann1 bash -c "
  cd /home/c00959374/AscendOpGenAgent
  python3 skills/ascendc/precision-tuning/scripts/precision_forensics.py \
    3_Add_pt1 --attempt 0 \
    --task-dir /home/c00959374/AscendOpGenAgent/outputs/codex_batch_20260420_1755/3_Add_pt1
"'
```

Expected: exit 0 + forensics_report_0.json written with non-null `error_distribution`, `worst_elements`, `tail_analysis`.

### Task 4.4: Run precision_gate to verify schema backward-compat

```bash
ssh npu_server 'docker exec cjm_cann1 bash -c "
  cd /home/c00959374/AscendOpGenAgent
  python3 skills/ascendc/precision-tuning/scripts/precision_gate.py \
    --step forensics --op-name 3_Add_pt1 --attempt 0 \
    --task-name outputs/codex_batch_20260420_1755/3_Add_pt1
"'
```

Expected: Gate-F PASS.

---

## Phase 5 — Parallel side tracks (independent of forensics)

### Task 5.1: Write STRUCTURE.md
**Files:** Create `skills/ascendc/precision-tuning/STRUCTURE.md` adapted from OpenOps-debug original, adjust paths to AscendOpGenAgent.

### Task 5.2: Diff agent files (OpenOps-debug vs AscendOpGenAgent)
**Files:** Modify `agents/precision-tuning.md` and `agents/precision-tuning-discovery.md` if original has content (other than anticheat section already added) that's missing in new version.

### Task 5.3: Final commit + push + remote pull

---

## Phase 6 — Wrap-up

- Delete `.planning/` transient docs (DESIGN_forensics_rewrite.md, task_plan.md) **only after** Phase 4 green
- Final status report to user

---

## Dependencies / Ordering

- Phase 1 tasks are independent internally (can be done in any order); Phase 2 depends on Phase 1; Phase 3 depends on 1 + 2; Phase 4 depends on 3; Phase 5 is independent of 1-4.
- Commits are atomic (one task = one commit) to keep revert clean.
- Remote NPU verification is the real gate; local numpy tests are smoke only.

## Remember
- Strict schema backward compat (Phase 3 must not break Gate-F)
- Parity test is mandatory before Phase 3
- Every phase independently revertable
- DRY, YAGNI — no speculative features beyond design doc
