#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _detect_ascend_path() -> Path:
    for env_name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()

    candidates = [
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[-1]


def _find_kernel_sources(kernel_dir: Path) -> list[Path]:
    sources = sorted(
        path for path in kernel_dir.glob("*.cpp")
        if path.name != "pybind11.cpp"
    )
    if not sources:
        raise FileNotFoundError(f"No kernel .cpp sources found in {kernel_dir}")
    return sources


def _extract_pybind_module_name(pybind_path: Path) -> str:
    content = pybind_path.read_text(encoding="utf-8")
    match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", content)
    if not match:
        raise ValueError(f"Unable to detect PYBIND11_MODULE name from {pybind_path}")
    return match.group(1)


def _pybind11_includes() -> str:
    """返回 `-I<python> -I<pybind11>` 形式的 include flags。

    优先级:
      1. torch 自带的 pybind11（`<torch>/include/pybind11/`）—— 与 libtorch 编译时
         使用的 pybind11 ABI 对齐，推荐用于 torch extension。
      2. pip 安装的 pybind11（`python3 -m pybind11 --includes`）—— fallback。

    返回的字符串用空格分隔，便于后续 CMake `string(REPLACE " " ";" ...)` 转为 list。
    """
    import sysconfig
    py_include = sysconfig.get_path("include")

    # 1) torch 自带 pybind11
    try:
        import torch  # type: ignore
        torch_include = Path(torch.__file__).resolve().parent / "include"
        if (torch_include / "pybind11" / "pybind11.h").is_file():
            return f"-I{py_include} -I{torch_include}"
    except ImportError:
        pass

    # 2) fallback: pip 装的 pybind11 module
    try:
        result = subprocess.run(
            ["python3", "-m", "pybind11", "--includes"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True,
        )
        flags = result.stdout.strip()
        if flags:
            return flags
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "Cannot locate pybind11 headers. Tried torch-bundled "
        "(<torch>/include/pybind11/) and `python3 -m pybind11 --includes`. "
        "Install pybind11 via `pip install pybind11`, or ensure torch is importable."
    )


def _format_cmake_list(items: list[str], indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{item}" for item in items)


def _generate_cmakelists(
    kernel_dir: Path,
    build_dir: Path,
    module_name: str,
    sources: list[Path],
    ascend_path: Path,
    pybind11_includes: str,
) -> str:
    include_dirs = [kernel_dir]
    catlass_include = kernel_dir / "catlass" / "include"
    if catlass_include.is_dir():
        include_dirs.append(catlass_include)

    task_catlass_include = kernel_dir.parent / "catlass" / "include"
    if task_catlass_include.is_dir() and task_catlass_include not in include_dirs:
        include_dirs.append(task_catlass_include)

    source_lines = [str(path) for path in sources]
    include_lines = [str(path) for path in include_dirs]

    return f"""cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${{SOC_VERSION}}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${{ASCEND_CANN_PACKAGE_PATH}}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "{build_dir}")

if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

ascendc_library(kernels STATIC
{_format_cmake_list(source_lines)}
)

ascendc_include_directories(kernels PRIVATE
{_format_cmake_list(include_lines)}
    ${{ASCEND_CANN_PACKAGE_PATH}}/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/runtime
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/msprof
)

add_library(pybind11_lib SHARED "{kernel_dir / 'pybind11.cpp'}")
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
  m
  dl
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${{TORCH_PATH}}")
set(ENV{{ASCEND_HOME_PATH}} ${{ASCEND_CANN_PACKAGE_PATH}})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${{TORCH_NPU_PATH}}")
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
target_include_directories(pybind11_lib PRIVATE
  "{kernel_dir}"
  ${{TORCH_NPU_PATH}}/include
  ${{TORCH_PATH}}/include
  ${{TORCH_PATH}}/include/torch/csrc/api/include
)
# PYBIND11 includes are precomputed by utils/build_ascendc.py at file-generation time.
# Priority: torch-bundled <torch>/include/pybind11/ (ABI-aligned with libtorch), then pip `python3 -m pybind11 --includes`.
set(PYBIND11_INC "{pybind11_includes}")
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {module_name}
  PREFIX ""
  SUFFIX "${{PYTHON_EXTENSION_SUFFIX}}"
)
"""


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[build_ascendc] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build(task: str, soc_version: str, build_type: str, clean: bool) -> Path:
    task_dir = _resolve_task_dir(task)
    kernel_dir = task_dir / "kernel"
    pybind_path = kernel_dir / "pybind11.cpp"
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")
    if not pybind_path.is_file():
        raise FileNotFoundError(f"Missing pybind11 entry: {pybind_path}")

    sources = _find_kernel_sources(kernel_dir)
    module_name = _extract_pybind_module_name(pybind_path)
    build_dir = kernel_dir / "build"
    cmake_dir = build_dir / "_autogen_cmake"
    ascend_path = _detect_ascend_path()
    pybind11_includes = _pybind11_includes()
    print(f"[build_ascendc] pybind11 includes: {pybind11_includes}")

    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmakelists_path = cmake_dir / "CMakeLists.txt"
    cmakelists_path.write_text(
        _generate_cmakelists(
            kernel_dir=kernel_dir,
            build_dir=build_dir,
            module_name=module_name,
            sources=sources,
            ascend_path=ascend_path,
            pybind11_includes=pybind11_includes,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(ascend_path)
    cmake_configure = [
        "cmake",
        "-S",
        str(cmake_dir),
        "-B",
        str(build_dir),
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    cmake_build = ["cmake", "--build", str(build_dir), "-j"]

    _run(cmake_configure, cwd=task_dir, env=env)
    _run(cmake_build, cwd=task_dir, env=env)
    return build_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AscendC kernels for a task without task-local run.sh")
    parser.add_argument("task", help="Task directory name or path")
    parser.add_argument("-v", "--soc-version", default="Ascend910B2", help="Ascend SoC version")
    parser.add_argument("--build-type", default="Debug", help="CMake build type")
    parser.add_argument("--clean", action="store_true", help="Remove kernel/build before configuring")
    args = parser.parse_args()

    build_dir = build(
        task=args.task,
        soc_version=args.soc_version,
        build_type=args.build_type,
        clean=args.clean,
    )
    print(f"[build_ascendc] Build completed: {build_dir}")


if __name__ == "__main__":
    main()
