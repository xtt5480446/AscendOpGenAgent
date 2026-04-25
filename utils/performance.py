import copy
import importlib.util
import inspect
import json as _json_mod
import os
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


def _benchmark_model(model, inputs, device, warmup: int, repeat: int):
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

                merged_latencies = []
                for index, inputs in enumerate(input_groups):
                    model_inputs = _move_to_device(_clone_value(inputs), device)
                    latencies = _benchmark_model(
                        model=model,
                        inputs=model_inputs,
                        device=device,
                        warmup=warmup,
                        repeat=repeat,
                    )
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

                result["latency_ms"] = merged_latencies
                result["mean_ms"] = statistics.mean(merged_latencies)
                result["median_ms"] = statistics.median(merged_latencies)
                result["min_ms"] = min(merged_latencies)
                result["max_ms"] = max(merged_latencies)
                result["stdev_ms"] = statistics.stdev(merged_latencies) if len(merged_latencies) > 1 else 0.0
                result["ok"] = True
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(f"{impl}: {result['error']}")

            report["results"].append(result)

        return report
    finally:
        if str(WORKDIR) in sys.path:
            sys.path.remove(str(WORKDIR))


def _print_report(report, summary_only=False):
    print("=" * 88)
    print("Performance Report")
    print("=" * 88)
    print(f"Operator  : {report['op']}")
    print(f"Task Dir  : {report['task_dir']}")
    print(f"Device    : {report['device']}")
    print(f"Warmup    : {report['warmup']}")
    print(f"Repeat    : {report['repeat']}")
    print(f"Seed      : {report['seed']}")

    if not summary_only and report["inputs"]:
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

    if not summary_only:
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


def _run_impl_isolated(op, impl, warmup, repeat, seed, json_out_path):
    """Run a single impl in an isolated subprocess, return parsed JSON report or None."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           op, impl, str(warmup), str(repeat), str(seed),
           "--json-out", json_out_path]
    subprocess.run(cmd)
    p = Path(json_out_path)
    if p.is_file():
        with open(p) as f:
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
    return base


def main():
    op, impls, warmup, repeat, seed, json_out = _parse_args(sys.argv)

    if len(impls) > 1:
        # Run each impl in an isolated subprocess so that a crash
        # (e.g. tilelang aicore exception) cannot taint later impls.
        tmp_dir = tempfile.mkdtemp(prefix="perf_iso_")
        sub_reports = []
        for impl in impls:
            tmp_json = os.path.join(tmp_dir, f"{impl}.json")
            sub_report = _run_impl_isolated(op, impl, warmup, repeat, seed, tmp_json)
            sub_reports.append(sub_report)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        report = _merge_reports(sub_reports)
        if report is None:
            print("All implementations failed.")
            raise SystemExit(1)
        _print_report(report, summary_only=True)
        if json_out:
            with open(json_out, "w") as f:
                _json_mod.dump(report, f, indent=2)
        if not any(r["ok"] for r in report["results"]):
            raise SystemExit(1)
        return

    report = _run_performance(op, impls, warmup, repeat, seed)
    _print_report(report)

    if json_out:
        with open(json_out, "w") as f:
            _json_mod.dump(report, f, indent=2)

    if not any(result["ok"] for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
