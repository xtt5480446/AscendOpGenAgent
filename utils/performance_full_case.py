"""
Full-case performance benchmark runner.

Usage:
    PERF_NPUS=auto python3 utils/performance_full_case.py <op> [impl] [warmup] [repeat] [seed]

Arguments:
    op      Task directory name or path.
    impl    reference | tilelang | ascendc | all (default: all)
    warmup  Warmup iterations (default: 5)
    repeat  Measurement iterations (default: 10)
    seed    Random seed (default: 0)

Environment:
    PERF_NPUS=auto
        Auto-detect available NPUs and run cases in parallel.
    PERF_NPUS=0,1,2,3
        Manually specify NPU IDs for parallel execution.

Behavior:
    - If multiple NPUs are available, cases are split across workers.
    - Each worker runs a subset of JSONL cases under its own ASCEND_RT_VISIBLE_DEVICES.
    - Results are merged into summary.csv and summary.md under the task directory.

Examples:
    PERF_NPUS=auto python3 utils/performance_full_case.py /home/Code/AscendOpGenAgent/archive_tasks/MhcPostGrad
    PERF_NPUS=0,1,2,3 python3 utils/performance_full_case.py MhcPostGrad ascendc 5 10 0
"""
import copy
import csv
import datetime
import importlib.util
import inspect
import json as _json_mod
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent

IMPLEMENTATIONS = {
    "reference": {
        "filename": "model.py",
        "preferred_class": "Model",
        "label": "Reference",
    },
    "tilelang": {
        "filename": "model_new_tilelang.py",
        "preferred_class": "ModelNew",
        "label": "TileLang",
    },
    "ascendc": {
        "filename": "model_new_ascendc.py",
        "preferred_class": "ModelNew",
        "label": "AscendC",
    },
}


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, preferred_name: str):
    candidate = getattr(module, preferred_name, None)
    if inspect.isclass(candidate) and issubclass(candidate, nn.Module):
        return candidate

    for _, value in vars(module).items():
        if inspect.isclass(value) and issubclass(value, nn.Module) and value is not nn.Module:
            return value

    raise AttributeError(f"No nn.Module subclass found in {module.__file__}")


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


CASE_TIMEOUT = int(os.environ.get("PERF_CASE_TIMEOUT", "300"))


class _CaseTimeout(Exception):
    pass


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _format_tensor_summary(tensor: torch.Tensor) -> str:
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"


def _summarize_value(value, name: str):
    if isinstance(value, torch.Tensor):
        return [f"{name}: {_format_tensor_summary(value)}"]
    if isinstance(value, list):
        lines = [f"{name}: list[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, tuple):
        lines = [f"{name}: tuple[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, dict):
        lines = [f"{name}: dict[{len(value)}]"]
        for key, item in value.items():
            lines.extend(_summarize_value(item, f"{name}.{key}"))
        return lines
    return [f"{name}: {type(value).__name__}({value})"]


def _get_input_groups(module):
    if not hasattr(module, "get_input_groups"):
        raise AttributeError(f"get_input_groups() not found in {module.__file__}")

    input_groups = module.get_input_groups()
    if not isinstance(input_groups, list) or not input_groups:
        raise ValueError("get_input_groups() must return a non-empty list")
    return input_groups


def _load_impl(task_dir: Path, op: str, impl: str):
    config = IMPLEMENTATIONS[impl]
    module_path = task_dir / config["filename"]
    if not module_path.is_file():
        raise FileNotFoundError(f"missing {impl} model: {module_path}")

    module = _load_module(module_path, f"{op}_{impl}_perf_model")
    model_cls = _find_model_class(module, config["preferred_class"])
    return module, model_cls, module_path


def _timeout_handler(signum, frame):
    raise _CaseTimeout("case timed out")


def _benchmark_model(model, inputs, device, warmup: int, repeat: int, timeout: int = 0):
    old_handler = None
    if timeout > 0 and hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(*inputs)
            _synchronize(device)

            timings_ms = []
            for _ in range(repeat):
                start = time.perf_counter()
                model(*inputs)
                _synchronize(device)
                end = time.perf_counter()
                timings_ms.append((end - start) * 1000.0)

        return timings_ms
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _run_performance(op: str, impls, warmup: int, repeat: int, seed: int):
    report = {
        "op": op,
        "device": str(_get_device()),
        "task_dir": "",
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "inputs": [],
        "results": [],
        "errors": [],
    }

    task_dir = _resolve_task_dir(op)
    report["task_dir"] = str(task_dir)
    device = _get_device()

    sys.path.insert(0, str(WORKDIR))
    try:
        ref_module = None
        init_inputs = []
        input_groups = []
        if (task_dir / "model.py").is_file():
            ref_module, _, _ = _load_impl(task_dir, op, "reference")
            init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()
            input_groups = _get_input_groups(ref_module)
        else:
            raise FileNotFoundError(f"missing reference model: {task_dir / 'model.py'}")

        input_summaries = []
        for index, inputs in enumerate(input_groups):
            input_summaries.extend(
                _summarize_value(_move_to_device(_clone_value(inputs), device), f"inputs[{index}]")
            )
        report["inputs"] = input_summaries

        total_cases = len(input_groups)
        for impl in impls:
            config = IMPLEMENTATIONS[impl]
            result = {
                "impl": impl,
                "label": config["label"],
                "model_path": "",
                "ok": False,
                "case_results": [],
                "latency_ms": [],
                "mean_ms": None,
                "median_ms": None,
                "min_ms": None,
                "max_ms": None,
                "stdev_ms": None,
                "error": "",
            }
            try:
                torch.manual_seed(seed)
                _, model_cls, module_path = _load_impl(task_dir, op, impl)
                model = model_cls(*_clone_value(init_inputs)).to(device).eval()
                result["model_path"] = str(module_path)

                print(f"\n[{config['label']}] {module_path}  ({total_cases} cases)", flush=True)

                merged_latencies = []
                timed_out_count = 0
                for index, inputs in enumerate(input_groups):
                    model_inputs = _move_to_device(_clone_value(inputs), device)
                    try:
                        latencies = _benchmark_model(
                            model=model,
                            inputs=model_inputs,
                            device=device,
                            warmup=warmup,
                            repeat=repeat,
                            timeout=CASE_TIMEOUT,
                        )
                    except _CaseTimeout:
                        timed_out_count += 1
                        print(
                            f"  case[{index}/{total_cases}] TIMEOUT (>{CASE_TIMEOUT}s)",
                            flush=True,
                        )
                        continue

                    case_result = {
                        "index": index,
                        "latency_ms": latencies,
                        "mean_ms": statistics.mean(latencies),
                        "median_ms": statistics.median(latencies),
                        "min_ms": min(latencies),
                        "max_ms": max(latencies),
                        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    }
                    result["case_results"].append(case_result)
                    merged_latencies.extend(latencies)
                    print(
                        f"  case[{index}/{total_cases}] mean={case_result['mean_ms']:.3f}ms "
                        f"median={case_result['median_ms']:.3f}ms "
                        f"min={case_result['min_ms']:.3f}ms max={case_result['max_ms']:.3f}ms",
                        flush=True,
                    )

                if merged_latencies:
                    result["latency_ms"] = merged_latencies
                    result["mean_ms"] = statistics.mean(merged_latencies)
                    result["median_ms"] = statistics.median(merged_latencies)
                    result["min_ms"] = min(merged_latencies)
                    result["max_ms"] = max(merged_latencies)
                    result["stdev_ms"] = statistics.stdev(merged_latencies) if len(merged_latencies) > 1 else 0.0
                    result["ok"] = True
                if timed_out_count:
                    result["timed_out"] = timed_out_count
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(f"{impl}: {result['error']}")

            report["results"].append(result)

        return report
    finally:
        if str(WORKDIR) in sys.path:
            sys.path.remove(str(WORKDIR))


def _detect_npus():
    """Detect available NPU IDs via npu-smi."""
    try:
        out = subprocess.check_output(["npu-smi", "info", "-l"], text=True, timeout=10)
        ids = []
        for line in out.splitlines():
            if line.strip().startswith("NPU ID"):
                parts = line.split(":")
                if len(parts) == 2:
                    ids.append(int(parts[1].strip()))
        return ids
    except Exception:
        return []


def _split_cases(jsonl_path: Path, n_splits: int):
    """Read JSONL and split into n roughly equal chunks."""
    lines = [l.strip() for l in open(jsonl_path) if l.strip()]
    if not lines:
        return []
    chunk_size = math.ceil(len(lines) / n_splits)
    return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]


def _run_parallel(op: str, impls, warmup: int, repeat: int, seed: int, npu_ids: list):
    """Run benchmarks in parallel across multiple NPUs, each handling a subset of cases."""
    task_dir = _resolve_task_dir(op)

    jsonl_files = [f for f in task_dir.glob("*.json") if not f.name.endswith(".bak")]
    if not jsonl_files:
        print("No .json case file found, falling back to single-NPU mode")
        return _run_performance(op, impls, warmup, repeat, seed)

    jsonl_path = jsonl_files[0]
    chunks = _split_cases(jsonl_path, len(npu_ids))
    actual_workers = len(chunks)

    if actual_workers <= 1:
        print("Too few cases to parallelize, falling back to single-NPU mode")
        return _run_performance(op, impls, warmup, repeat, seed)

    total = sum(len(c) for c in chunks)
    print(f"Parallelizing {total} cases across {actual_workers} NPUs: "
          f"{npu_ids[:actual_workers]}", flush=True)

    tmp_base = tempfile.mkdtemp(prefix="perf_parallel_")
    processes = []
    json_out_files = []

    impl_arg = "all" if set(impls) == set(IMPLEMENTATIONS.keys()) else (
        impls[0] if len(impls) == 1 else impls[0])

    try:
        offset = 0
        for i, chunk in enumerate(chunks):
            npu_id = npu_ids[i]
            worker_dir = Path(tmp_base) / f"npu{npu_id}"
            shutil.copytree(task_dir, worker_dir)

            with open(worker_dir / jsonl_path.name, "w") as f:
                for line in chunk:
                    f.write(line + "\n")

            json_out = Path(tmp_base) / f"result_npu{npu_id}.json"
            json_out_files.append(json_out)

            print(f"  NPU {npu_id}: case[{offset}..{offset + len(chunk) - 1}] "
                  f"({len(chunk)} cases)", flush=True)
            offset += len(chunk)

            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
            env["PERF_CASE_TIMEOUT"] = str(CASE_TIMEOUT)
            env.pop("PERF_NPUS", None)  # prevent recursive parallel

            cmd = [
                sys.executable, str(SCRIPT_DIR / "performance_full_case.py"),
                str(worker_dir), impl_arg,
                str(warmup), str(repeat), str(seed),
                "--json-out", str(json_out),
            ]
            proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
            processes.append((npu_id, proc))

        for npu_id, proc in processes:
            proc.wait()
            if proc.returncode != 0:
                print(f"  NPU {npu_id} worker exited with code {proc.returncode}",
                      flush=True)

        return _merge_parallel_results(op, task_dir, json_out_files, npu_ids[:actual_workers],
                                        warmup, repeat, seed)
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)


def _merge_parallel_results(op, task_dir, json_out_files, npu_ids, warmup, repeat, seed):
    """Merge JSON reports from parallel workers."""
    merged = {
        "op": op,
        "device": "npu (parallel)",
        "task_dir": str(task_dir),
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "inputs": [],
        "results": [],
        "errors": [],
    }

    worker_reports = []
    impl_results = {}
    global_offset = 0
    for i, jf in enumerate(json_out_files):
        if not jf.exists():
            continue
        with open(jf) as f:
            report = _json_mod.load(f)
        worker_reports.append((npu_ids[i], report))
        merged["errors"].extend(report.get("errors", []))

        # Find max local case count in this worker to compute offset
        local_case_count = 0
        for result in report.get("results", []):
            local_case_count = max(local_case_count, len(result.get("case_results", [])))

        for result in report.get("results", []):
            impl = result["impl"]
            if impl not in impl_results:
                impl_results[impl] = {
                    "impl": impl,
                    "label": result["label"],
                    "model_path": result.get("model_path", ""),
                    "ok": False,
                    "case_results": [],
                    "latency_ms": [],
                    "mean_ms": None, "median_ms": None,
                    "min_ms": None, "max_ms": None, "stdev_ms": None,
                    "error": "",
                    "timed_out": 0,
                }
            mr = impl_results[impl]
            for cr in result.get("case_results", []):
                cr["index"] = cr["index"] + global_offset
            mr["case_results"].extend(result.get("case_results", []))
            mr["latency_ms"].extend(result.get("latency_ms", []))
            mr["timed_out"] += result.get("timed_out", 0)
            if result.get("error"):
                mr["error"] = result["error"]

        global_offset += local_case_count

    for mr in impl_results.values():
        lats = mr["latency_ms"]
        if lats:
            mr["mean_ms"] = statistics.mean(lats)
            mr["median_ms"] = statistics.median(lats)
            mr["min_ms"] = min(lats)
            mr["max_ms"] = max(lats)
            mr["stdev_ms"] = statistics.stdev(lats) if len(lats) > 1 else 0.0
            mr["ok"] = True
        if not mr["timed_out"]:
            mr.pop("timed_out", None)
        merged["results"].append(mr)

    merged["_worker_reports"] = worker_reports
    return merged


def _write_csv_and_summary(report, task_dir: Path, worker_reports=None):
    """Write summary.csv, summary.md, and per-group CSVs to perf_YYYYMMDD_HHMMSS/"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_dir = task_dir / f"perf_{timestamp}"
    perf_dir.mkdir(exist_ok=True)
    groups_dir = perf_dir / "groups"
    groups_dir.mkdir(exist_ok=True)

    # Build case-level data
    impl_map = {r["impl"]: r for r in report["results"]}
    case_rows = []
    for impl_name, impl_result in impl_map.items():
        for cr in impl_result.get("case_results", []):
            case_rows.append({
                "impl": impl_name,
                "case_index": cr["index"],
                "mean_ms": cr["mean_ms"],
                "median_ms": cr["median_ms"],
                "min_ms": cr["min_ms"],
                "max_ms": cr["max_ms"],
                "stdev_ms": cr["stdev_ms"],
            })

    # Group by case_index
    case_index_map = {}
    for row in case_rows:
        idx = row["case_index"]
        if idx not in case_index_map:
            case_index_map[idx] = {}
        case_index_map[idx][row["impl"]] = row

    # Write summary.csv
    summary_csv = perf_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_index",
            "reference_mean_ms", "reference_median_ms", "reference_min_ms", "reference_max_ms", "reference_stdev_ms",
            "tilelang_mean_ms", "tilelang_median_ms", "tilelang_min_ms", "tilelang_max_ms", "tilelang_stdev_ms",
            "ascendc_mean_ms", "ascendc_median_ms", "ascendc_min_ms", "ascendc_max_ms", "ascendc_stdev_ms",
            "ascendc_vs_ref_speedup", "tilelang_vs_ref_speedup"
        ])
        for idx in sorted(case_index_map.keys()):
            impls_data = case_index_map[idx]
            ref = impls_data.get("reference", {})
            tl = impls_data.get("tilelang", {})
            ac = impls_data.get("ascendc", {})
            ref_mean = ref.get("mean_ms")
            tl_mean = tl.get("mean_ms")
            ac_mean = ac.get("mean_ms")
            ac_speedup = ref_mean / ac_mean if ref_mean and ac_mean else None
            tl_speedup = ref_mean / tl_mean if ref_mean and tl_mean else None
            writer.writerow([
                idx,
                ref.get("mean_ms", ""), ref.get("median_ms", ""), ref.get("min_ms", ""), ref.get("max_ms", ""), ref.get("stdev_ms", ""),
                tl.get("mean_ms", ""), tl.get("median_ms", ""), tl.get("min_ms", ""), tl.get("max_ms", ""), tl.get("stdev_ms", ""),
                ac.get("mean_ms", ""), ac.get("median_ms", ""), ac.get("min_ms", ""), ac.get("max_ms", ""), ac.get("stdev_ms", ""),
                f"{ac_speedup:.3f}" if ac_speedup else "",
                f"{tl_speedup:.3f}" if tl_speedup else "",
            ])

    # Write summary.md
    summary_md = perf_dir / "summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# Performance Summary\n\n")
        f.write(f"**Operator**: {report['op']}\n\n")
        f.write(f"**Task Dir**: {report['task_dir']}\n\n")
        f.write(f"**Device**: {report['device']}\n\n")
        f.write(f"**Warmup**: {report['warmup']}, **Repeat**: {report['repeat']}, **Seed**: {report['seed']}\n\n")
        f.write(f"**Timestamp**: {timestamp}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Overall Statistics\n\n")
        f.write(f"| Impl | Status | Mean(ms) | Median(ms) | Min(ms) | Max(ms) | Std(ms) | Timeout |\n")
        f.write(f"|------|--------|----------|------------|---------|---------|---------|----------|\n")
        for r in report["results"]:
            status = "OK" if r["ok"] else "ERROR"
            timeout_str = f"{r.get('timed_out', 0)}" if r.get("timed_out") else "-"
            if r["ok"]:
                f.write(f"| {r['impl']} | {status} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | {r['min_ms']:.3f} | {r['max_ms']:.3f} | {r['stdev_ms']:.3f} | {timeout_str} |\n")
            else:
                f.write(f"| {r['impl']} | {status} | - | - | - | - | - | {timeout_str} |\n")
        f.write(f"\n")

        # Speedup summary
        ref_result = impl_map.get("reference")
        ac_result = impl_map.get("ascendc")
        tl_result = impl_map.get("tilelang")
        if ref_result and ref_result["ok"] and ac_result and ac_result["ok"]:
            ac_speedup = ref_result["mean_ms"] / ac_result["mean_ms"]
            f.write(f"**AscendC vs Reference Speedup**: {ac_speedup:.3f}x\n\n")
        if ref_result and ref_result["ok"] and tl_result and tl_result["ok"]:
            tl_speedup = ref_result["mean_ms"] / tl_result["mean_ms"]
            f.write(f"**TileLang vs Reference Speedup**: {tl_speedup:.3f}x\n\n")

        f.write(f"---\n\n")
        f.write(f"**Files**:\n\n")
        f.write(f"- `summary.csv`: per-case detailed results\n")
        f.write(f"- `summary.md`: this file\n")
        if worker_reports:
            f.write(f"- `groups/`: per-NPU group CSVs\n")

    # Write per-group CSVs if worker_reports provided
    if worker_reports:
        # Group all worker reports by npu_id so each CSV contains all impls
        grouped = {}
        for npu_id, worker_report in worker_reports:
            grouped.setdefault(npu_id, []).append(worker_report)
        for npu_id in sorted(grouped.keys()):
            group_csv = groups_dir / f"npu{npu_id}.csv"
            with open(group_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["case_index", "impl", "mean_ms", "median_ms", "min_ms", "max_ms", "stdev_ms"])
                for worker_report in grouped[npu_id]:
                    for r in worker_report.get("results", []):
                        for cr in r.get("case_results", []):
                            writer.writerow([cr["index"], r["impl"], cr["mean_ms"], cr["median_ms"], cr["min_ms"], cr["max_ms"], cr["stdev_ms"]])

    return perf_dir


def _print_report(report):
    print("=" * 88)
    print("Performance Report")
    print("=" * 88)
    print(f"Operator  : {report['op']}")
    print(f"Task Dir  : {report['task_dir']}")
    print(f"Device    : {report['device']}")
    print(f"Warmup    : {report['warmup']}")
    print(f"Repeat    : {report['repeat']}")
    print(f"Seed      : {report['seed']}")

    if report["inputs"]:
        print("-" * 88)
        print("Inputs")
        print("-" * 88)
        for line in report["inputs"]:
            print(line)

    print("-" * 88)
    print(f"{'Impl':<12} {'Status':<8} {'Mean(ms)':>12} {'Median':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    print("-" * 88)
    for result in report["results"]:
        if result["ok"]:
            print(
                f"{result['impl']:<12} {'OK':<8} "
                f"{result['mean_ms']:>12.3f} {result['median_ms']:>12.3f} "
                f"{result['min_ms']:>12.3f} {result['max_ms']:>12.3f} "
                f"{result['stdev_ms']:>12.3f}"
            )
        else:
            print(f"{result['impl']:<12} {'ERROR':<8} {'-':>12} {'-':>12} {'-':>12} {'-':>12} {'-':>12}")
        if result.get("timed_out"):
            print(f"  ({result['timed_out']} case(s) timed out, excluded from stats)")

    for result in report["results"]:
        print("-" * 88)
        print(f"{result['impl']} -> {result['model_path'] or 'N/A'}")
        if result["ok"]:
            for case_result in result["case_results"]:
                samples = ", ".join(f"{value:.3f}" for value in case_result["latency_ms"])
                print(
                    f"case[{case_result['index']}] mean={case_result['mean_ms']:.3f} ms, "
                    f"median={case_result['median_ms']:.3f} ms, samples(ms): [{samples}]"
                )
        else:
            print(f"error      : {result['error']}")


def _parse_args(argv):
    json_out = None
    filtered = []
    i = 1
    while i < len(argv):
        if argv[i] == "--json-out" and i + 1 < len(argv):
            json_out = argv[i + 1]
            i += 2
        else:
            filtered.append(argv[i])
            i += 1

    if len(filtered) < 1 or len(filtered) > 5:
        print("Usage: python utils/performance.py <op> [impl] [warmup] [repeat] [seed]")
        print("  impl: reference | tilelang | ascendc | all")
        print("  env PERF_NPUS=0,1,2,3 to parallelize across NPUs")
        raise SystemExit(1)

    op = filtered[0]
    impl = filtered[1] if len(filtered) >= 2 else "all"
    warmup = int(filtered[2]) if len(filtered) >= 3 else 5
    repeat = int(filtered[3]) if len(filtered) >= 4 else 10
    seed = int(filtered[4]) if len(filtered) >= 5 else 0

    if impl == "all":
        impls = ["reference", "tilelang", "ascendc"]
    elif impl in IMPLEMENTATIONS:
        impls = [impl]
    else:
        raise SystemExit(f"Unsupported impl '{impl}'")

    if warmup < 0 or repeat <= 0:
        raise SystemExit("warmup must be >= 0 and repeat must be > 0")

    return op, impls, warmup, repeat, seed, json_out


def _run_impl_isolated(op, impl, warmup, repeat, seed, npu_ids, json_out_path):
    """Run a single impl in an isolated subprocess, return parsed JSON report or None."""
    cmd = [sys.executable, str(Path(__file__).resolve()), op, impl,
           str(warmup), str(repeat), str(seed), "--json-out", json_out_path]
    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env)
    if Path(json_out_path).is_file():
        with open(json_out_path) as f:
            return _json_mod.load(f)
    return None


def _merge_reports(reports):
    """Merge per-impl reports into a single combined report."""
    base = None
    for r in reports:
        if r is None:
            continue
        if base is None:
            base = r
            continue
        base["results"].extend(r.get("results", []))
        base["errors"].extend(r.get("errors", []))
        wr = r.pop("_worker_reports", None)
        if wr:
            base.setdefault("_worker_reports", []).extend(wr)
    return base


def main():
    op, impls, warmup, repeat, seed, json_out = _parse_args(sys.argv)

    npu_env = os.environ.get("PERF_NPUS", "")
    if npu_env:
        if npu_env.strip() == "auto":
            npu_ids = _detect_npus()
        else:
            npu_ids = [int(x.strip()) for x in npu_env.split(",") if x.strip()]
    else:
        npu_ids = []

    if len(impls) > 1:
        # Run each impl in an isolated subprocess so that a crash
        # (e.g. tilelang aicore exception) cannot taint later impls.
        tmp_dir = tempfile.mkdtemp(prefix="perf_iso_")
        sub_reports = []
        for impl in impls:
            tmp_json = os.path.join(tmp_dir, f"{impl}.json")
            sub_report = _run_impl_isolated(op, impl, warmup, repeat, seed, npu_ids, tmp_json)
            sub_reports.append(sub_report)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        report = _merge_reports(sub_reports)
        if report is None:
            print("All implementations failed.")
            raise SystemExit(1)
    else:
        if len(npu_ids) > 1:
            report = _run_parallel(op, impls, warmup, repeat, seed, npu_ids)
        else:
            report = _run_performance(op, impls, warmup, repeat, seed)

    # If invoked as a subprocess for isolated impl, only write JSON and exit
    if json_out and len(impls) == 1:
        with open(json_out, "w") as f:
            _json_mod.dump(report, f, indent=2)
        if not any(result["ok"] for result in report["results"]):
            raise SystemExit(1)
        return

    # Write CSV and summary
    task_dir = Path(report["task_dir"])
    worker_reports = report.pop("_worker_reports", None)
    perf_dir = _write_csv_and_summary(report, task_dir, worker_reports)

    # Print summary
    print("\n" + "=" * 88)
    print("Performance Summary")
    print("=" * 88)
    print(f"Operator  : {report['op']}")
    print(f"Task Dir  : {report['task_dir']}")
    print(f"Device    : {report['device']}")
    print(f"Output    : {perf_dir}")
    print("-" * 88)
    print(f"{'Impl':<12} {'Status':<8} {'Mean(ms)':>12} {'Median':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    print("-" * 88)
    for result in report["results"]:
        if result["ok"]:
            print(
                f"{result['impl']:<12} {'OK':<8} "
                f"{result['mean_ms']:>12.3f} {result['median_ms']:>12.3f} "
                f"{result['min_ms']:>12.3f} {result['max_ms']:>12.3f} "
                f"{result['stdev_ms']:>12.3f}"
            )
        else:
            print(f"{result['impl']:<12} {'ERROR':<8} {'-':>12} {'-':>12} {'-':>12} {'-':>12} {'-':>12}")
        if result.get("timed_out"):
            print(f"  ({result['timed_out']} case(s) timed out)")

    # Speedup
    impl_map = {r["impl"]: r for r in report["results"]}
    ref = impl_map.get("reference")
    ac = impl_map.get("ascendc")
    tl = impl_map.get("tilelang")
    if ref and ref["ok"] and ac and ac["ok"]:
        speedup = ref["mean_ms"] / ac["mean_ms"]
        print(f"\nAscendC vs Reference Speedup: {speedup:.3f}x")
    if ref and ref["ok"] and tl and tl["ok"]:
        speedup = ref["mean_ms"] / tl["mean_ms"]
        print(f"TileLang vs Reference Speedup: {speedup:.3f}x")

    print("=" * 88)
    print(f"\nResults written to: {perf_dir}/")
    print(f"  - summary.csv")
    print(f"  - summary.md")
    if worker_reports:
        print(f"  - groups/ ({len(worker_reports)} NPU group CSVs)")

    if json_out:
        with open(json_out, "w") as f:
            _json_mod.dump(report, f, indent=2)

    if not any(result["ok"] for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
