#!/usr/bin/env python3
"""AscendC 算子精度调试分析模板。

使用方式：
    复制到 {task_dir}/debug_{op_name}_precision.py，替换占位符 {{OP_NAME}} /
    {{TASK_DIR}} / {{CASE_INDEX}} 后，通过 references/run_precision_debug.sh 运行。

与 `utils/verification_ascendc.py` 共用 Model 加载/输入生成/张量比对逻辑，本模板
只在其上叠加「误差分布分析层」（首错坐标、错误周期、特殊值检测、错误百分比），
对应 SKILL.md Phase 1 产物。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
# 由 run_precision_debug.sh 保证 WORKDIR 环境变量指向 repo 根；本脚本作为兜底
# 向上回溯，找到包含 utils/verification_ascendc.py 的目录。
def _find_workdir() -> Path:
    env = os.environ.get("WORKDIR")
    if env:
        candidate = Path(env).resolve()
        if (candidate / "utils" / "verification_ascendc.py").is_file():
            return candidate
    cursor = SCRIPT_DIR
    for _ in range(10):
        if (cursor / "utils" / "verification_ascendc.py").is_file():
            return cursor
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    raise RuntimeError("cannot locate repo root containing utils/verification_ascendc.py")


WORKDIR = _find_workdir()
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utils.verification_ascendc import (  # noqa: E402
    _clone_value,
    _find_model_class,
    _get_device,
    _get_input_groups,
    _load_module,
    _move_to_device,
    _normalize_output,
)


# ========== 配置区 — 由 agent/skill 替换 ==========
OP_NAME: str = "{{OP_NAME}}"
TASK_DIR: str = "{{TASK_DIR}}"   # 相对 WORKDIR 或绝对路径
CASE_INDEX: int = {{CASE_INDEX}}
# ====================================================


THRESHOLD_BY_DTYPE = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float32: 1e-5,
    torch.float64: 1e-5,
}


def _resolve_task_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.is_dir():
        return p
    candidate = (WORKDIR / raw).resolve()
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"task_dir not found: {raw}")


def _count_tensors(value) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(_count_tensors(v) for v in value)
    if isinstance(value, dict):
        return sum(_count_tensors(v) for v in value.values())
    return 0


def _is_single_tensor_inputs(inputs) -> bool:
    return len(inputs) == 1 and _count_tensors(inputs[0]) == 1


def _iter_tensor_pairs(ref, cand, path: str = "output"):
    """同步递归 ref/cand，产出 (path, ref_tensor, cand_tensor) 或 (path, mismatch_reason)."""
    if type(ref) is not type(cand):
        yield path, f"type mismatch: ref={type(ref).__name__}, cand={type(cand).__name__}"
        return
    if isinstance(ref, torch.Tensor):
        yield path, ref, cand
        return
    if isinstance(ref, (list, tuple)):
        if len(ref) != len(cand):
            yield path, f"length mismatch: ref={len(ref)}, cand={len(cand)}"
            return
        for i, (a, b) in enumerate(zip(ref, cand)):
            yield from _iter_tensor_pairs(a, b, f"{path}[{i}]")
        return
    if isinstance(ref, dict):
        if ref.keys() != cand.keys():
            yield path, f"dict keys mismatch"
            return
        for k in ref:
            yield from _iter_tensor_pairs(ref[k], cand[k], f"{path}.{k}")
        return
    # 标量
    if ref != cand:
        yield path, f"scalar mismatch: ref={ref}, cand={cand}"


def _analyze_pair(path: str, ref: torch.Tensor, cand: torch.Tensor, tag: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {tag}  {path}")
    print(f"  shape(ref)={tuple(ref.shape)} dtype(ref)={ref.dtype}")
    print(f"  shape(cand)={tuple(cand.shape)} dtype(cand)={cand.dtype}")
    print(f"{'=' * 60}")

    if ref.shape != cand.shape:
        print(f"  !! shape mismatch — skip numeric analysis (非精度问题)")
        return

    ref_fp = torch.nan_to_num(ref.to(torch.float32))
    cand_fp = torch.nan_to_num(cand.to(torch.float32))
    abs_err = (ref_fp - cand_fp).abs()
    rel_err = abs_err / ref_fp.abs().clamp(min=1e-12)

    max_abs = abs_err.max().item() if abs_err.numel() else 0.0
    mean_abs = abs_err.mean().item() if abs_err.numel() else 0.0
    max_rel = rel_err.max().item() if rel_err.numel() else 0.0
    mean_rel = rel_err.mean().item() if rel_err.numel() else 0.0
    print(f"  MaxAbsErr : {max_abs:.6e}")
    print(f"  MeanAbsErr: {mean_abs:.6e}")
    print(f"  MaxRelErr : {max_rel:.6e}")
    print(f"  MeanRelErr: {mean_rel:.6e}")

    if abs_err.numel() and ref_fp.numel() > 1:
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_fp.flatten().unsqueeze(0), cand_fp.flatten().unsqueeze(0)
        ).item()
        print(f"  CosineSim : {cos_sim:.8f}")

    all_zero_cand = cand_fp.numel() > 0 and (cand_fp == 0).all().item()
    has_nan_cand = torch.isnan(cand.to(torch.float32)).any().item()
    has_inf_cand = torch.isinf(cand.to(torch.float32)).any().item()
    if all_zero_cand:
        print("  [warn] cand 输出全零")
    if has_nan_cand:
        print("  [warn] cand 输出含 NaN")
    if has_inf_cand:
        print("  [warn] cand 输出含 Inf")

    thr = THRESHOLD_BY_DTYPE.get(cand.dtype, 1e-2)
    fail_mask = abs_err > thr
    total = abs_err.numel()
    fail_count = int(fail_mask.sum().item()) if total else 0
    if fail_count == 0:
        print(f"  OK 所有元素均在阈值 {thr:g} 内")
        return

    print(f"  错误元素: {fail_count}/{total} ({100 * fail_count / total:.4f}%)")

    flat_fails = fail_mask.flatten().nonzero(as_tuple=False).squeeze(-1)
    first_linear = int(flat_fails[0].item())
    # 反推多维坐标
    coord = []
    rem = first_linear
    for d in range(ref.ndim - 1, -1, -1):
        size = ref.shape[d]
        coord.append(int(rem % size))
        rem //= size
    coord = tuple(reversed(coord))
    ref_val = ref_fp.flatten()[first_linear].item()
    cand_val = cand_fp.flatten()[first_linear].item()
    err_val = abs_err.flatten()[first_linear].item()
    print(
        f"  首错线性下标: {first_linear}  多维坐标: {coord}\n"
        f"    REF={ref_val:.6e}  CAND={cand_val:.6e}  AbsErr={err_val:.6e}"
    )

    # 周期性检测
    if flat_fails.numel() > 2:
        diffs = flat_fails[1:] - flat_fails[:-1]
        uniq = diffs.unique()
        if uniq.numel() <= 5:
            print(f"  错误间隔: {uniq.tolist()}  (可能存在周期性)")


def _run_one_case(ref_model: nn.Module, cand_model: nn.Module, inputs, device, tag: str) -> None:
    ref_inputs = _move_to_device(_clone_value(inputs), device)
    cand_inputs = _move_to_device(_clone_value(inputs), device)
    with torch.no_grad():
        ref_out = ref_model(*ref_inputs)
        cand_out = cand_model(*cand_inputs)
    ref_out = _normalize_output(ref_out)
    cand_out = _normalize_output(cand_out)

    pair_count = 0
    for item in _iter_tensor_pairs(ref_out, cand_out):
        if len(item) == 2:
            path, reason = item
            print(f"\n  [{tag}] {path}: {reason}  (非数值精度问题)")
            continue
        path, r, c = item
        _analyze_pair(path, r, c, tag=tag)
        pair_count += 1
    if pair_count == 0:
        print(f"  [{tag}] 未提取到可比对的 tensor 输出")


def _make_fixed_input(template: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "ones":
        return torch.ones_like(template)
    if mode == "arange":
        n = template.numel()
        flat = torch.arange(n, dtype=torch.float32, device=template.device)
        flat = flat / max(n - 1, 1)  # 0..1
        return flat.reshape(template.shape).to(template.dtype)
    raise ValueError(mode)


def _shrink_inputs(template: torch.Tensor, size: int) -> torch.Tensor:
    # 生成与 template dtype 一致、长度为 size 的随机张量（1D）
    gen = torch.randn(size, dtype=torch.float32, device=template.device)
    lo = template.to(torch.float32).min().item() if template.numel() else 0.0
    hi = template.to(torch.float32).max().item() if template.numel() else 1.0
    span = hi - lo if hi > lo else 1.0
    scaled = gen * span * 0.5 + (lo + hi) * 0.5
    return scaled.to(template.dtype)


def main() -> int:
    print(f"===== {OP_NAME} 精度调试  (case_index={CASE_INDEX}) =====")
    task_dir = _resolve_task_dir(TASK_DIR)
    ref_path = task_dir / "model.py"
    cand_path = task_dir / "model_new_ascendc.py"
    kernel_build = task_dir / "kernel" / "build"
    print(f"TaskDir  : {task_dir}")
    print(f"Ref      : {ref_path}")
    print(f"Cand     : {cand_path}")
    if not ref_path.is_file() or not cand_path.is_file():
        print("!! missing model.py or model_new_ascendc.py")
        return 2
    if kernel_build.is_dir() and str(kernel_build) not in sys.path:
        sys.path.insert(0, str(kernel_build))

    ref_module = _load_module(ref_path, f"{OP_NAME}_ref_dbg")
    cand_module = _load_module(cand_path, f"{OP_NAME}_cand_dbg")
    ref_cls = _find_model_class(ref_module, "Model")
    cand_cls = _find_model_class(cand_module, "ModelNew")

    torch.manual_seed(0)
    if hasattr(cand_module, "get_init_inputs"):
        init_inputs = cand_module.get_init_inputs()
    else:
        init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()

    input_groups = _get_input_groups(ref_module)
    if not (0 <= CASE_INDEX < len(input_groups)):
        print(f"!! CASE_INDEX={CASE_INDEX} 超出范围 [0, {len(input_groups)})")
        return 2
    inputs = input_groups[CASE_INDEX]

    device = _get_device()
    print(f"Device   : {device}")
    ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
    cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()

    # ---------- 1. 原失败 case ----------
    print(f"\n----- [1] 复现失败 case #{CASE_INDEX} (原始输入) -----")
    _run_one_case(ref_model, cand_model, inputs, device, tag="原始输入")

    # ---------- 2. 固定输入对照（仅单 tensor 输入）----------
    if _is_single_tensor_inputs(inputs):
        template = inputs[0]
        for mode in ("ones", "arange"):
            try:
                print(f"\n----- [2.{mode}] 固定输入对照 -----")
                fixed = _make_fixed_input(template, mode)
                _run_one_case(ref_model, cand_model, [fixed], device, tag=f"{mode}输入")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] mode={mode} 失败: {e}")

        # ---------- 3. 缩小 shape 二分 ----------
        if template.ndim >= 1:
            for size in (32, 256, 1024, 4096):
                if size >= template.numel():
                    continue
                try:
                    print(f"\n----- [3.shrink={size}] 缩小 shape 二分 -----")
                    small = _shrink_inputs(template, size)
                    _run_one_case(ref_model, cand_model, [small], device, tag=f"shape=({size},)")
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] size={size} 失败: {e}")
    else:
        print("\n[info] 多输入/嵌套输出 case，跳过 固定输入对照 / 缩小 shape 二分")

    print("\n===== 分析完成 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
