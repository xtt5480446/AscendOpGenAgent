#!/usr/bin/env python3
"""
_forensics_child.py — ascendc-debug forensics 的子进程 tensor executor。

由 precision_forensics.py 主进程 spawn (subprocess.run) 调用，只做一件事：
  1. 加载 {task_dir}/model.py 和 {task_dir}/model_new_ascendc.py
  2. 对 get_input_groups() 所有 case 分别跑 ref / cand 模型
  3. 把 ref_out / cand_out / inputs / int8_triggered / atol / rtol 通过 pickle
     存到 {dump_dir}/case_{N}.pkl + {dump_dir}/metadata.json
  4. 每个 case 跑完立即 del tensors 释放 NPU 内存

为什么 copy 而不是 import utils/verification_ascendc.py:
  - 子进程隔离避免主进程 sys.path / torch state / extension 污染
  - 允许 forensics 独立演化（加 dump hook、中间 tensor probe 等）
    而不牵动 bench 评测链路

但是**语义必须与 utils/verification_ascendc.py 完全一致**:
  - 模型加载顺序：Model / ModelNew fallback 到第一个 nn.Module 子类
  - init_inputs 优先级：cand_module.get_init_inputs() 优先
  - 输入：get_input_groups() 优先，fallback [get_inputs()]
  - 容差默认 atol=1e-2, rtol=1e-2
  - int8 特判：ref_out 与 cand_out 都含 int8 tensor 时切 atol=1.5, rtol=0.0
  - 种子 torch.manual_seed(0)
  - 设备选择 _get_device() 的 npu/cuda/cpu 优先级

下方复制区（`# ── BEGIN COPIED FROM utils/verification_ascendc.py ──` ...
`# ── END ──`）是逐字复制。若 utils/verification_ascendc.py 的对应函数发生
语义变化，必须同步更新这里（或通过 parity test 捕获漂移）。

用法:
  python3 _forensics_child.py <task_dir> <dump_dir>

退出码:
  0  全部 case 成功
  1  运行时错误（详情在 stderr）
  2  参数错误
"""

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # AscendOpGenAgent/


# ── BEGIN COPIED FROM utils/verification_ascendc.py (lines 17-247) ──
# 若修改此区, 必须同步更新 utils/verification_ascendc.py 或通过
# test_executor_parity.py 捕获不一致。禁止"只改一处"。

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


def _normalize_output(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_normalize_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_output(item) for key, item in value.items()}
    return value


def _contains_int8_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.dtype == torch.int8
    if isinstance(value, list):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, tuple):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_int8_tensor(item) for item in value.values())
    return False


def _get_input_groups(module):
    if hasattr(module, "get_input_groups"):
        input_groups = module.get_input_groups()
        if not isinstance(input_groups, list) or not input_groups:
            raise ValueError("get_input_groups() must return a non-empty list")
        return input_groups

    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("get_inputs() must return a non-empty list")
        return [inputs]

    raise AttributeError(
        f"Neither get_input_groups() nor get_inputs() found in {module.__file__}"
    )


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── END COPIED FROM utils/verification_ascendc.py ──


DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
INT8_ATOL = 1.5
INT8_RTOL = 0.0


def _npu_sync():
    """If running on NPU, synchronize and free cache. No-op elsewhere."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def _compute_loader_hash() -> str:
    """Hash of the copied loader region, embedded in metadata so parity tests
    and forensics readers can detect a drift between this file and
    utils/verification_ascendc.py if we ever change one side."""
    src = Path(__file__).read_text(encoding="utf-8")
    begin = src.find("# ── BEGIN COPIED FROM")
    end = src.find("# ── END COPIED FROM")
    region = src[begin:end] if begin != -1 and end != -1 else src
    return hashlib.sha256(region.encode("utf-8")).hexdigest()[:16]


def run(task_dir: str, dump_dir: str) -> int:
    task_dir_p = Path(task_dir).resolve()
    dump_dir_p = Path(dump_dir).resolve()
    dump_dir_p.mkdir(parents=True, exist_ok=True)

    ref_path = task_dir_p / "model.py"
    cand_path = task_dir_p / "model_new_ascendc.py"
    kernel_build = task_dir_p / "kernel" / "build"

    if not ref_path.is_file():
        print(f"[forensics-child] missing reference model: {ref_path}", file=sys.stderr)
        return 1
    if not cand_path.is_file():
        print(f"[forensics-child] missing candidate model: {cand_path}", file=sys.stderr)
        return 1

    # sys.path injection — same order as utils/verification_ascendc.py
    inserted_paths = []
    paths_to_add = [str(REPO_ROOT)]
    if kernel_build.is_dir():
        paths_to_add.append(str(kernel_build))
    else:
        # kernel/build 不存在时照样尝试 import（model_new_ascendc.py 可能自行
        # sys.path.insert），只是打印警告而不报错
        print(
            f"[forensics-child] WARN: {kernel_build} not found; "
            f"assuming model_new_ascendc.py manages its own import path",
            file=sys.stderr,
        )
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted_paths.append(p)

    num_cases = 0
    device_str = ""
    try:
        op_tag = task_dir_p.name  # e.g. "3_Add_pt1"
        ref_module = _load_module(ref_path, f"{op_tag}_forensics_ref_model")
        cand_module = _load_module(cand_path, f"{op_tag}_forensics_cand_model")

        ref_cls = _find_model_class(ref_module, "Model")
        cand_cls = _find_model_class(cand_module, "ModelNew")

        torch.manual_seed(0)

        # cand 优先（允许 ModelNew 覆盖 Model 的 init 参数，与 verification 一致）
        if hasattr(cand_module, "get_init_inputs"):
            init_inputs = cand_module.get_init_inputs()
        else:
            init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()

        input_groups = _get_input_groups(ref_module)
        num_cases = len(input_groups)

        device = _get_device()
        device_str = str(device)

        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
        cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()

        for case_idx, inputs in enumerate(input_groups):
            ref_inputs_dev = _move_to_device(_clone_value(inputs), device)
            cand_inputs_dev = _move_to_device(_clone_value(inputs), device)
            raw_inputs_cpu = _normalize_output(_clone_value(inputs))

            with torch.no_grad():
                ref_out = ref_model(*ref_inputs_dev)
                cand_out = cand_model(*cand_inputs_dev)
            _npu_sync()

            ref_out_cpu = _normalize_output(ref_out)
            cand_out_cpu = _normalize_output(cand_out)

            int8_triggered = (
                _contains_int8_tensor(ref_out_cpu)
                and _contains_int8_tensor(cand_out_cpu)
            )
            atol = INT8_ATOL if int8_triggered else DEFAULT_ATOL
            rtol = INT8_RTOL if int8_triggered else DEFAULT_RTOL

            case_payload = {
                "case_idx": case_idx,
                "ref": ref_out_cpu,
                "cand": cand_out_cpu,
                "inputs": raw_inputs_cpu,
                "int8_triggered": int8_triggered,
                "atol": atol,
                "rtol": rtol,
            }
            with (dump_dir_p / f"case_{case_idx}.pkl").open("wb") as f:
                pickle.dump(case_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            del ref_inputs_dev, cand_inputs_dev, ref_out, cand_out
            del ref_out_cpu, cand_out_cpu, raw_inputs_cpu, case_payload
            _npu_sync()

        metadata = {
            "status": "completed",
            "num_cases": num_cases,
            "device": device_str,
            "atol_default": DEFAULT_ATOL,
            "rtol_default": DEFAULT_RTOL,
            "int8_atol": INT8_ATOL,
            "int8_rtol": INT8_RTOL,
            "loader_parity_hash": _compute_loader_hash(),
            "task_dir": str(task_dir_p),
        }
        (dump_dir_p / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    except Exception as exc:
        err_meta = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "num_cases_attempted": num_cases,
            "device": device_str,
            "loader_parity_hash": _compute_loader_hash(),
            "task_dir": str(task_dir_p),
        }
        (dump_dir_p / "metadata.json").write_text(
            json.dumps(err_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[forensics-child] ERROR: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        for p in inserted_paths:
            if p in sys.path:
                sys.path.remove(p)


def main():
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <task_dir> <dump_dir>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(run(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
