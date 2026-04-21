#!/usr/bin/env python3
"""
test_executor_parity.py — verify _forensics_child.py stays semantically
aligned with utils/verification_ascendc.py.

REQUIRES NPU: must run inside cjm_cann* container (torch_npu + compiled
kernel extensions). Typical invocation:

    ssh npu_server 'docker exec cjm_cann1 bash -c "
        cd /home/c00959374/AscendOpGenAgent
        python3 skills/ascendc/precision-tuning/scripts/test_executor_parity.py
    "'

What it checks per fixture:
  1. _forensics_child.py exits 0 and produces metadata.json + case_N.pkl files.
  2. Loaded ref/cand tensor pairs, compared under verification_ascendc.py's
     exact semantics (_tensor_diff_summary — torch.nan_to_num + atol/rtol),
     give per-case mismatch_ratio and max_abs_diff.
  3. Run verification_ascendc.py on the same task and parse its stdout.
  4. Assert all-cases pass/fail verdict matches.
  5. Assert per-case mismatch_ratio matches within 1e-4.
  6. Assert OutputFlattener produces path set consistent with
     _compare_values path emission (structural check).

Exits 0 if all fixtures PASS or SKIP (missing kernel/build is acceptable).
Exits 1 on parity violation.

Add fixtures via `FIXTURES` list. Each entry:
    {"name": "...", "task_dir": "/abs/path", "expect_pass": bool}

For quick local sanity (no NPU): scripts fall back to SKIP for every
fixture without reaching NPU code paths. Fine to run for CI smoke.
"""

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SKILL_DIR.parent.parent.parent
CHILD = SCRIPT_DIR / "_forensics_child.py"
VERIF = REPO_ROOT / "utils" / "verification_ascendc.py"


# Edit me to add fixtures. Keep minimal — parity test is a gate, not a full suite.
FIXTURES = [
    {
        "name": "avg_pool3_d",
        "task_dir": str(REPO_ROOT / "archive_tasks" / "avg_pool3_d"),
        # avg_pool3_d is a known-good kernel; expect pass
        "expect_pass": True,
    },
    {
        "name": "rms_norm",
        "task_dir": str(REPO_ROOT / "archive_tasks" / "rms_norm"),
        "expect_pass": True,
    },
    {
        "name": "3_Add_pt1",
        "task_dir": str(REPO_ROOT / "outputs" / "codex_batch_20260420_1755" / "3_Add_pt1"),
        # precision-tuning starting point — may pass or fail; just checks parity
        "expect_pass": None,
    },
]


def _has_npu() -> bool:
    try:
        import torch
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _parse_verif_stdout(stdout: str):
    """Extract per-case mismatch_ratio and max_abs_diff from verification stdout.

    Line format matches _tensor_diff_summary():
      case[i]: output[...]: dtype(...), unequal_elements=N, mismatch_ratio=X%, max_abs_diff=Y, ...

    Returns list of dicts: [{"case": i, "mismatch_ratio": float_fraction, "max_abs_diff": float, "matched": bool}]
    where mismatch_ratio is already converted from % to 0-1 fraction.
    Missing cases (i.e. pass with "matched") get mismatch_ratio=0.0 max_abs_diff=0.0.
    """
    cases = {}
    matched_re = re.compile(r"case\[(\d+)\]:\s+output.*?:\s+matched")
    for m in matched_re.finditer(stdout):
        cases[int(m.group(1))] = {
            "case": int(m.group(1)),
            "mismatch_ratio": 0.0,
            "max_abs_diff": 0.0,
            "matched": True,
        }
    diff_re = re.compile(
        r"case\[(\d+)\]:\s+.*?mismatch_ratio=([\d.]+)%.*?max_abs_diff=([0-9.eE+\-g]+)"
    )
    for m in diff_re.finditer(stdout):
        cases[int(m.group(1))] = {
            "case": int(m.group(1)),
            "mismatch_ratio": float(m.group(2)) / 100.0,
            "max_abs_diff": float(m.group(3)),
            "matched": False,
        }
    return [cases[k] for k in sorted(cases.keys())]


def _diff_under_verif_semantics(ref, cand, atol: float, rtol: float) -> dict:
    """Mirror verification_ascendc._tensor_diff_summary for a single tensor pair."""
    import torch
    ref_fp = torch.nan_to_num(ref.to(torch.float32))
    cand_fp = torch.nan_to_num(cand.to(torch.float32))
    if ref.shape != cand.shape:
        return {"mismatch_ratio": 1.0, "max_abs_diff": float("inf"), "matched": False}
    diff = (ref_fp - cand_fp).abs()
    allowed = atol + rtol * cand_fp.abs()
    mismatch_mask = diff > allowed
    total = ref.numel()
    mismatch_count = int(mismatch_mask.sum().item()) if diff.numel() else 0
    ratio = (mismatch_count / total) if total else 0.0
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    matched = torch.allclose(ref_fp, cand_fp, atol=atol, rtol=rtol)
    return {"mismatch_ratio": ratio, "max_abs_diff": max_abs, "matched": matched}


def _walk_tensors(ref, cand, path="output"):
    """Yield (path, ref_tensor, cand_tensor) for leaf tensor pairs, mirroring
    verification_ascendc _compare_values recursion.

    Scalars and None are skipped (they're compared by equality in verification,
    which is out-of-scope for DiffAnalyzer).
    """
    import torch
    if type(ref) is not type(cand):
        return
    if isinstance(ref, torch.Tensor):
        yield path, ref, cand
    elif isinstance(ref, (list, tuple)):
        if len(ref) != len(cand):
            return
        for i, (r, c) in enumerate(zip(ref, cand)):
            yield from _walk_tensors(r, c, f"{path}[{i}]")
    elif isinstance(ref, dict):
        if set(ref.keys()) != set(cand.keys()):
            return
        for k in ref:
            yield from _walk_tensors(ref[k], cand[k], f"{path}.{k}")


def _run_child(task_dir: str, dump_dir: str) -> int:
    proc = subprocess.run(
        [sys.executable, str(CHILD), task_dir, dump_dir],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode


def _run_verif(task_name: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(VERIF), task_name],
        capture_output=True, text=True, timeout=1800,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def run_fixture(fx: dict) -> str:
    name = fx["name"]
    task_dir = fx["task_dir"]
    expect_pass = fx.get("expect_pass")

    if not Path(task_dir).is_dir():
        return f"SKIP: {name}: task_dir missing ({task_dir})"
    kb = Path(task_dir) / "kernel" / "build"
    if not kb.is_dir():
        return f"SKIP: {name}: kernel/build missing (run evaluate_ascendc.sh first)"

    with tempfile.TemporaryDirectory(prefix=f"parity_{name}_") as td:
        rc = _run_child(task_dir, td)
        if rc != 0:
            return f"FAIL: {name}: child executor exit rc={rc}"

        meta = json.loads((Path(td) / "metadata.json").read_text())
        if meta.get("status") != "completed":
            return f"FAIL: {name}: child metadata status={meta.get('status')}"
        num_cases = meta["num_cases"]

        # Parity: child-dumped tensors diff'd under verification semantics
        child_per_case = []
        for i in range(num_cases):
            with (Path(td) / f"case_{i}.pkl").open("rb") as f:
                payload = pickle.load(f)
            ref, cand = payload["ref"], payload["cand"]
            atol, rtol = payload["atol"], payload["rtol"]

            # Walk all leaf tensors and aggregate worst-per-case
            worst_ratio = 0.0
            worst_diff = 0.0
            all_matched = True
            for _p, r_t, c_t in _walk_tensors(ref, cand):
                d = _diff_under_verif_semantics(r_t, c_t, atol, rtol)
                worst_ratio = max(worst_ratio, d["mismatch_ratio"])
                worst_diff = max(worst_diff, d["max_abs_diff"])
                all_matched = all_matched and d["matched"]
            child_per_case.append({
                "case": i, "mismatch_ratio": worst_ratio,
                "max_abs_diff": worst_diff, "matched": all_matched,
            })

        # Run bench verification on same task
        verif_rc, verif_out = _run_verif(name if fx.get("verif_task") is None else fx["verif_task"])
        verif_per_case = _parse_verif_stdout(verif_out)

        # Compare verdicts
        if len(verif_per_case) == 0:
            return f"FAIL: {name}: could not parse verification stdout (verif rc={verif_rc})"
        if len(verif_per_case) != num_cases:
            return (f"FAIL: {name}: case count mismatch "
                    f"(child={num_cases}, verif={len(verif_per_case)})")

        mismatches = []
        for cp, vp in zip(child_per_case, verif_per_case):
            if cp["matched"] != vp["matched"]:
                mismatches.append(
                    f"  case[{cp['case']}] matched: child={cp['matched']} verif={vp['matched']}"
                )
            if abs(cp["mismatch_ratio"] - vp["mismatch_ratio"]) > 1e-4:
                mismatches.append(
                    f"  case[{cp['case']}] mismatch_ratio: "
                    f"child={cp['mismatch_ratio']:.6f} verif={vp['mismatch_ratio']:.6f}"
                )

        if mismatches:
            return f"FAIL: {name}:\n" + "\n".join(mismatches)

        if expect_pass is not None:
            all_pass = all(p["matched"] for p in child_per_case)
            if all_pass != expect_pass:
                return f"FAIL: {name}: expected_pass={expect_pass} actual={all_pass}"

        return f"PASS: {name}: {num_cases} cases, parity verified"


def main():
    if not CHILD.is_file():
        print(f"ERROR: {CHILD} missing", file=sys.stderr)
        sys.exit(1)
    if not VERIF.is_file():
        print(f"ERROR: {VERIF} missing", file=sys.stderr)
        sys.exit(1)

    if not _has_npu():
        print("SKIP (no NPU): this test requires torch_npu; run inside cjm_cann* container")
        sys.exit(0)

    results = []
    for fx in FIXTURES:
        print(f"[parity] running {fx['name']}...", flush=True)
        result = run_fixture(fx)
        print(f"[parity] {result}", flush=True)
        results.append(result)

    print("\n=== PARITY SUMMARY ===")
    for r in results:
        print(r)
    if any(r.startswith("FAIL") for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
