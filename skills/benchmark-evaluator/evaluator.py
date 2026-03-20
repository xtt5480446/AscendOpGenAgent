#!/usr/bin/env python3
"""
Benchmark Evaluator - 串行执行评测框架
调用用户指定的 Agent 在 KernelBench 数据集上进行代码生成能力评测
"""

import os
import sys
import json
import time
import re
import glob
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import subprocess
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """任务配置"""
    agent_name: str
    agent_workspace: str
    level_problems: Dict[int, Optional[Any]]  # {level: [problem_ids] or None}
    benchmark_path: str = "/mnt/w00934874/agent/benchmark/KernelBench"
    output_root: str = "./benchmark_results"
    arch: str = "ascend910b1"
    resume: bool = True
    timeout_per_task: int = 2400
    warmup: int = 5
    repeats: int = 50


@dataclass
class GenerationResult:
    """代码生成结果"""
    success: bool = False
    code: str = ""
    generation_time: float = 0.0
    error_message: str = ""


@dataclass
class VerifyResult:
    """验证结果"""
    success: bool = False
    compiled: bool = False
    correctness: bool = False
    max_diff: Optional[float] = None
    error_message: str = ""


@dataclass
class PerformanceResult:
    """性能测试结果"""
    success: bool = False
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    speedup_vs_torch: float = 0.0
    error_message: str = ""


@dataclass
class EvaluationResult:
    """完整评测结果"""
    level: int
    problem_id: int
    op_name: str
    op_type: str
    timestamp: str = ""
    generation: Optional[GenerationResult] = None
    verification: Optional[VerifyResult] = None
    performance: Optional[PerformanceResult] = None
    output_dir: str = ""


class AgentLoader:
    """Agent 加载器"""
    
    @staticmethod
    def load_agent_config(agent_workspace: str, agent_name: str) -> Dict[str, Any]:
        """从工作区加载 Agent 配置，支持 primary 和 subagent"""
        agent_file = os.path.join(agent_workspace, "agents", f"{agent_name}.md")
        
        if not os.path.exists(agent_file):
            raise ValueError(f"Agent 文件不存在: {agent_file}")
        
        # 读取文件内容
        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析 frontmatter
        config = AgentLoader._parse_frontmatter(content)
        
        config['name'] = agent_name
        config['workspace'] = agent_workspace
        config['file'] = agent_file
        
        return config
    
    @staticmethod
    def _parse_frontmatter(content: str) -> Dict[str, Any]:
        """解析 markdown frontmatter"""
        config = {}
        
        # 匹配 frontmatter 模式
        pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            frontmatter = match.group(1)
            try:
                # 尝试使用 YAML 解析
                config = yaml.safe_load(frontmatter) or {}
            except Exception as e:
                logger.warning(f"解析 frontmatter 失败: {e}")
        
        return config
    
    @staticmethod
    def get_agent_skills(agent_config: Dict[str, Any]) -> List[str]:
        """获取 Agent 的 skills 列表"""
        skills = agent_config.get('skills', [])
        if isinstance(skills, str):
            skills = [skills]
        return skills or []


class TaskScanner:
    """任务扫描器"""
    
    @staticmethod
    def parse_problem_ids(problem_ids: Optional[Any]) -> Optional[List[int]]:
        """解析 problem_ids 参数"""
        if problem_ids is None:
            return None
        
        if isinstance(problem_ids, list):
            return problem_ids
        
        if isinstance(problem_ids, str):
            # 支持 "1-10" 或 "1,2,3" 格式
            result = []
            parts = problem_ids.split(',')
            for part in parts:
                if '-' in part:
                    start, end = part.split('-')
                    result.extend(range(int(start), int(end) + 1))
                else:
                    result.append(int(part))
            return result
        
        return None
    
    @staticmethod
    def scan_tasks(
        benchmark_path: str,
        level_problems: Dict[int, Optional[Any]]
    ) -> List[Dict[str, Any]]:
        """扫描任务
        
        Args:
            benchmark_path: Benchmark 根目录
            level_problems: {level: [problem_ids] or None}
        
        Returns:
            任务列表
        """
        tasks = []
        
        for level, problem_ids in level_problems.items():
            level_dir = os.path.join(benchmark_path, f"level{level}")
            
            if not os.path.exists(level_dir):
                logger.warning(f"Level 目录不存在: {level_dir}")
                continue
            
            # 解析 problem_ids
            parsed_ids = TaskScanner.parse_problem_ids(problem_ids)
            
            # 扫描任务文件
            task_files = glob.glob(os.path.join(level_dir, "*.py"))
            
            for task_file in task_files:
                # 提取 problem_id
                match = re.match(r'(\d+)_.*\.py', os.path.basename(task_file))
                if not match:
                    continue
                
                pid = int(match.group(1))
                
                # 过滤 problem_ids
                if parsed_ids is not None and pid not in parsed_ids:
                    continue
                
                # 提取算子名
                op_name = match.group(0).replace('.py', '').split('_', 1)[1]
                
                tasks.append({
                    'level': level,
                    'problem_id': pid,
                    'task_file': task_file,
                    'op_name': op_name
                })
        
        # 排序
        tasks.sort(key=lambda x: (x['level'], x['problem_id']))
        
        return tasks
    
    @staticmethod
    def classify_op_type(op_name: str) -> str:
        """分类算子类型"""
        op_name_lower = op_name.lower()
        
        if any(kw in op_name_lower for kw in ['matmul', 'bmm', 'linear', 'gemm']):
            return 'matmul'
        elif any(kw in op_name_lower for kw in ['conv']):
            return 'conv'
        elif any(kw in op_name_lower for kw in ['softmax', 'layernorm', 'batchnorm', 'sum', 'mean', 'max', 'min']):
            return 'reduce'
        elif any(kw in op_name_lower for kw in ['attention', 'mha', 'sdpa']):
            return 'attention'
        elif any(kw in op_name_lower for kw in ['add', 'mul', 'sub', 'div', 'relu', 'sigmoid', 'tanh', 'gelu', 'silu']):
            return 'elementwise'
        else:
            return 'other'


class StateManager:
    """状态管理器（断点续跑）"""
    
    def __init__(self, output_dir: str):
        self.state_file = os.path.join(output_dir, ".eval_state.json")
        self.completed_tasks = set()
        self.load_state()
    
    def load_state(self):
        """加载状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.completed_tasks = set(tuple(x) for x in data.get('completed', []))
                logger.info(f"已加载 {len(self.completed_tasks)} 个已完成任务")
            except Exception as e:
                logger.warning(f"加载状态失败: {e}")
    
    def save_state(self):
        """保存状态"""
        try:
            data = {'completed': [list(x) for x in self.completed_tasks]}
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")
    
    def is_completed(self, level: int, problem_id: int) -> bool:
        """检查任务是否已完成"""
        return (level, problem_id) in self.completed_tasks
    
    def mark_completed(self, level: int, problem_id: int):
        """标记任务完成"""
        self.completed_tasks.add((level, problem_id))
        self.save_state()


class AgentCaller:
    """Agent 调用器 - 直接调用 kernelgen-workflow subagent"""
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.agent_config = agent_config
        self.agent_name = agent_config['name']
        self.workspace = agent_config['workspace']
    
    def generate_code(self, task_file: str, arch: str, output_dir: str, timeout: int = 2400) -> GenerationResult:
        """直接调用 kernelgen-workflow subagent 生成代码并验证"""
        result = GenerationResult()
        start_time = time.time()
        
        try:
            # 构建调用 kernelgen-workflow subagent 的 prompt
            # 直接传递所有必要参数，无需交互
            prompt = f"""生成并验证算子代码。

任务文件路径: {task_file}
输出路径: {output_dir}
arch: {arch}
框架: torch
后端: ascend
DSL: triton_ascend
warmup: 5
repeats: 50

请直接执行生成和验证流程，无需用户确认。"""
            
            logger.info(f"调用 kernelgen-workflow subagent 生成代码...")
            logger.info(f"Task file: {task_file}")
            logger.info(f"Arch: {arch}")
            logger.info(f"Output dir: {output_dir}")
            
            # 使用 opencode run 直接调用 kernelgen-workflow
            cmd = [
                "opencode", "run",
                "--agent", "kernelgen-workflow",
                prompt
            ]
            
            logger.info(f"执行命令: {' '.join(cmd[:4])}...")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            
            # 记录输出
            if proc.stdout:
                logger.info(f"Agent stdout:\n{proc.stdout[:1000]}...")  # 截断长输出
            if proc.stderr:
                logger.warning(f"Agent stderr:\n{proc.stderr[:1000]}...")
            
            # 检查返回码
            if proc.returncode != 0:
                result.success = False
                result.error_message = f"Agent 执行失败，返回码: {proc.returncode}\nstderr: {proc.stderr}\nstdout: {proc.stdout}"
                logger.error(result.error_message)
                return result
            
            # 检查生成的代码文件
            generated_code_file = os.path.join(output_dir, "generated_code.py")
            
            if os.path.exists(generated_code_file):
                with open(generated_code_file, 'r') as f:
                    result.code = f.read()
                result.success = True
                logger.info(f"代码生成成功，文件: {generated_code_file}")
            else:
                result.success = False
                result.error_message = f"代码生成失败，未找到输出文件: {generated_code_file}"
                logger.error(result.error_message)
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = f"代码生成超时（{timeout}秒）"
            logger.error(result.error_message)
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"代码生成失败: {e}")
        
        result.generation_time = time.time() - start_time
        return result


class VerifierCaller:
    """验证器调用器"""
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.agent_config = agent_config
        self.workspace = agent_config['workspace']
        
        # 获取 kernel-verifier skill 路径
        skills = AgentLoader.get_agent_skills(agent_config)
        verifier_skill = None
        for skill in skills:
            if 'verifier' in skill.lower():
                verifier_skill = skill
                break
        
        if verifier_skill:
            self.scripts_dir = os.path.join(self.workspace, "skills", verifier_skill, "scripts")
        else:
            # 默认路径
            self.scripts_dir = os.path.join(self.workspace, "skills", "kernel-verifier", "scripts")
    
    def verify(
        self,
        op_name: str,
        verify_dir: str,
        timeout: int = 300
    ) -> VerifyResult:
        """调用验证脚本"""
        result = VerifyResult()
        
        try:
            verify_script = os.path.join(self.scripts_dir, "verify.py")
            
            if not os.path.exists(verify_script):
                logger.warning(f"验证脚本不存在: {verify_script}")
                result.success = False
                result.error_message = "验证脚本不存在"
                return result
            
            cmd = [
                "python3", verify_script,
                "--op_name", op_name,
                "--verify_dir", verify_dir,
                "--timeout", str(timeout)
            ]
            
            logger.info(f"执行验证: {' '.join(cmd)}")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if proc.returncode == 0 and "验证成功" in proc.stdout:
                result.success = True
                result.compiled = True
                result.correctness = True
            else:
                result.success = False
                result.error_message = proc.stderr or proc.stdout
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = f"验证超时（{timeout}秒）"
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def benchmark(
        self,
        op_name: str,
        verify_dir: str,
        output_file: str,
        warmup: int = 5,
        repeats: int = 50,
        timeout: int = 300
    ) -> PerformanceResult:
        """调用性能测试脚本"""
        result = PerformanceResult()
        
        try:
            benchmark_script = os.path.join(self.scripts_dir, "benchmark.py")
            
            if not os.path.exists(benchmark_script):
                logger.warning(f"性能测试脚本不存在: {benchmark_script}")
                result.success = False
                result.error_message = "性能测试脚本不存在"
                return result
            
            cmd = [
                "python3", benchmark_script,
                "--op_name", op_name,
                "--verify_dir", verify_dir,
                "--warmup", str(warmup),
                "--repeats", str(repeats),
                "--output", output_file
            ]
            
            logger.info(f"执行性能测试: {' '.join(cmd)}")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if proc.returncode == 0 and os.path.exists(output_file):
                # 读取性能结果
                with open(output_file, 'r') as f:
                    perf_data = json.load(f)
                
                result.success = True
                result.avg_latency_ms = perf_data.get('implementation', {}).get('avg_latency_ms', 0)
                result.p50_latency_ms = perf_data.get('implementation', {}).get('p50_latency_ms', 0)
                result.p99_latency_ms = perf_data.get('implementation', {}).get('p99_latency_ms', 0)
                result.peak_memory_mb = perf_data.get('implementation', {}).get('peak_memory_mb', 0)
                result.speedup_vs_torch = perf_data.get('speedup_vs_torch', 0)
            else:
                result.success = False
                result.error_message = proc.stderr or proc.stdout
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = f"性能测试超时（{timeout}秒）"
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result


def evaluate_single_task(
    task_info: Dict[str, Any],
    agent_config: Dict[str, Any],
    config: TaskConfig,
    output_base_dir: str
) -> EvaluationResult:
    """评测单个任务"""
    
    level = task_info['level']
    problem_id = task_info['problem_id']
    task_file = task_info['task_file']
    op_name = task_info['op_name']
    
    # 分类算子类型
    op_type = TaskScanner.classify_op_type(op_name)
    
    # 创建输出目录
    task_output_dir = os.path.join(
        output_base_dir,
        f"level_{level}",
        f"{problem_id}_{op_name}"
    )
    os.makedirs(task_output_dir, exist_ok=True)
    
    # 初始化结果
    result = EvaluationResult(
        level=level,
        problem_id=problem_id,
        op_name=op_name,
        op_type=op_type,
        timestamp=datetime.now().isoformat(),
        output_dir=task_output_dir
    )
    
    try:
        # 读取任务描述
        with open(task_file, 'r') as f:
            task_desc = f.read()
        
        # 1. 代码生成
        logger.info(f"[Level {level} Problem {problem_id}]: 生成代码...")
        agent_caller = AgentCaller(agent_config)
        
        gen_result = agent_caller.generate_code(
            task_file=task_file,
            arch=config.arch,
            output_dir=task_output_dir,
            timeout=config.timeout_per_task
        )
        result.generation = gen_result
        
        if not gen_result.success:
            logger.error(f"代码生成失败: {gen_result.error_message}")
            return result
        
        # 生成的代码已由 agent 保存到 task_output_dir/generated_code.py
        generated_code_file = os.path.join(task_output_dir, "generated_code.py")
        
        # 2. 准备验证环境
        verify_dir = os.path.join(task_output_dir, "verify")
        os.makedirs(verify_dir, exist_ok=True)
        
        # 复制任务文件和生成的代码到验证目录
        torch_file = os.path.join(verify_dir, f"{op_name}_torch.py")
        impl_file = os.path.join(verify_dir, f"{op_name}_triton_ascend_impl.py")
        
        with open(torch_file, 'w') as f:
            f.write(task_desc)
        
        # 从生成的代码文件复制到验证目录
        if os.path.exists(generated_code_file):
            with open(generated_code_file, 'r') as f:
                generated_code = f.read()
            with open(impl_file, 'w') as f:
                f.write(generated_code)
        else:
            logger.error(f"生成的代码文件不存在: {generated_code_file}")
            result.generation.success = False
            result.generation.error_message = f"生成的代码文件不存在: {generated_code_file}"
            return result
        
        # 3. 正确性验证
        logger.info(f"[Level {level} Problem {problem_id}]: 验证正确性...")
        verifier = VerifierCaller(agent_config)
        verify_result = verifier.verify(
            op_name=op_name,
            verify_dir=verify_dir,
            timeout=config.timeout_per_task
        )
        result.verification = verify_result
        
        # 保存验证结果
        verify_result_file = os.path.join(task_output_dir, "verify_result.json")
        with open(verify_result_file, 'w') as f:
            json.dump(asdict(verify_result), f, indent=2)
        
        if not verify_result.success:
            logger.warning(f"验证失败: {verify_result.error_message}")
            return result
        
        # 4. 性能测试
        logger.info(f"[Level {level} Problem {problem_id}]: 性能测试...")
        perf_output_file = os.path.join(task_output_dir, "perf_result.json")
        perf_result = verifier.benchmark(
            op_name=op_name,
            verify_dir=verify_dir,
            output_file=perf_output_file,
            warmup=config.warmup,
            repeats=config.repeats,
            timeout=config.timeout_per_task
        )
        result.performance = perf_result
        
        logger.info(
            f"[Level {level} Problem {problem_id}]: "
            f"完成 (编译: {verify_result.compiled}, "
            f"正确: {verify_result.correctness}, "
            f"加速比: {perf_result.speedup_vs_torch:.2f}x)"
        )
        
    except Exception as e:
        logger.error(f"评测异常: {e}")
        logger.error(traceback.format_exc())
        result.generation = result.generation or GenerationResult()
        result.generation.error_message = str(e)
    
    return result


class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def generate_agent_report(
        agent_name: str,
        results: List[EvaluationResult],
        output_file: str
    ):
        """生成单个 Agent 报告"""
        
        # 统计计算
        total = len(results)
        compiled = sum(1 for r in results if r.verification and r.verification.compiled)
        correct = sum(1 for r in results if r.verification and r.verification.correctness)
        
        speedups = [
            r.performance.speedup_vs_torch 
            for r in results 
            if r.performance and r.performance.success
        ]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        
        # 按 Level 统计（包含优于PyTorch的比例）
        level_stats = {}
        for level in [1, 2, 3, 4]:
            level_results = [r for r in results if r.level == level]
            if level_results:
                level_compiled = sum(1 for r in level_results if r.verification and r.verification.compiled)
                level_correct = sum(1 for r in level_results if r.verification and r.verification.correctness)
                level_speedups = [
                    r.performance.speedup_vs_torch 
                    for r in level_results 
                    if r.performance and r.performance.success
                ]
                level_avg_speedup = sum(level_speedups) / len(level_speedups) if level_speedups else 0
                # 统计优于PyTorch的数量（speedup > 1.0）
                better_than_torch = sum(1 for s in level_speedups if s > 1.0)
                better_ratio = better_than_torch / len(level_speedups) * 100 if level_speedups else 0
                
                level_stats[level] = {
                    'total': len(level_results),
                    'compiled': level_compiled,
                    'correct': level_correct,
                    'speedups': level_speedups,
                    'avg_speedup': level_avg_speedup,
                    'better_than_torch': better_than_torch,
                    'better_ratio': better_ratio
                }
        
        # 按算子类型统计
        type_stats = {}
        for result in results:
            op_type = result.op_type
            if op_type not in type_stats:
                type_stats[op_type] = {'total': 0, 'compiled': 0, 'correct': 0, 'speedups': []}
            type_stats[op_type]['total'] += 1
            if result.verification:
                if result.verification.compiled:
                    type_stats[op_type]['compiled'] += 1
                if result.verification.correctness:
                    type_stats[op_type]['correct'] += 1
            if result.performance and result.performance.success:
                type_stats[op_type]['speedups'].append(result.performance.speedup_vs_torch)
        
        # 收集失败案例
        compile_failed = [r for r in results if r.verification and not r.verification.compiled]
        verify_failed = [r for r in results if r.verification and r.verification.compiled and not r.verification.correctness]
        perf_degraded = [r for r in results if r.performance and r.performance.success and r.performance.speedup_vs_torch < 1.0]
        
        # 生成 Markdown 报告
        report = f"""# Agent: {agent_name} 评测报告

## 执行摘要
- 执行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 总任务数: {total}

## 总体统计
| 指标 | Level1 | Level2 | Level3 | Level4 | 总体 |
|------|--------|--------|--------|--------|------|
| 任务数 | {level_stats.get(1, {}).get('total', 0)} | {level_stats.get(2, {}).get('total', 0)} | {level_stats.get(3, {}).get('total', 0)} | {level_stats.get(4, {}).get('total', 0)} | {total} |
| 编译成功 | {level_stats.get(1, {}).get('compiled', 0)} ({level_stats.get(1, {}).get('compiled', 0)/level_stats.get(1, {}).get('total', 1)*100:.0f}%) | {level_stats.get(2, {}).get('compiled', 0)} ({level_stats.get(2, {}).get('compiled', 0)/level_stats.get(2, {}).get('total', 1)*100:.0f}%) | {level_stats.get(3, {}).get('compiled', 0)} ({level_stats.get(3, {}).get('compiled', 0)/level_stats.get(3, {}).get('total', 1)*100:.0f}%) | {level_stats.get(4, {}).get('compiled', 0)} ({level_stats.get(4, {}).get('compiled', 0)/level_stats.get(4, {}).get('total', 1)*100:.0f}%) | {compiled} ({compiled/total*100:.0f}%) |
| 数值正确 | {level_stats.get(1, {}).get('correct', 0)} ({level_stats.get(1, {}).get('correct', 0)/level_stats.get(1, {}).get('total', 1)*100:.0f}%) | {level_stats.get(2, {}).get('correct', 0)} ({level_stats.get(2, {}).get('correct', 0)/level_stats.get(2, {}).get('total', 1)*100:.0f}%) | {level_stats.get(3, {}).get('correct', 0)} ({level_stats.get(3, {}).get('correct', 0)/level_stats.get(3, {}).get('total', 1)*100:.0f}%) | {level_stats.get(4, {}).get('correct', 0)} ({level_stats.get(4, {}).get('correct', 0)/level_stats.get(4, {}).get('total', 1)*100:.0f}%) | {correct} ({correct/total*100:.0f}%) |
| 优于PyTorch | {level_stats.get(1, {}).get('better_than_torch', 0)} ({level_stats.get(1, {}).get('better_ratio', 0):.0f}%) | {level_stats.get(2, {}).get('better_than_torch', 0)} ({level_stats.get(2, {}).get('better_ratio', 0):.0f}%) | {level_stats.get(3, {}).get('better_than_torch', 0)} ({level_stats.get(3, {}).get('better_ratio', 0):.0f}%) | {level_stats.get(4, {}).get('better_than_torch', 0)} ({level_stats.get(4, {}).get('better_ratio', 0):.0f}%) | {sum(s > 1.0 for s in speedups)} ({sum(s > 1.0 for s in speedups)/len(speedups)*100 if speedups else 0:.0f}%) |
| 平均加速比 | {level_stats.get(1, {}).get('avg_speedup', 0):.2f}x | {level_stats.get(2, {}).get('avg_speedup', 0):.2f}x | {level_stats.get(3, {}).get('avg_speedup', 0):.2f}x | {level_stats.get(4, {}).get('avg_speedup', 0):.2f}x | {avg_speedup:.2f}x |

## 按算子类型统计
| 类型 | 任务数 | 编译成功 | 数值正确 | 平均加速比 |
|------|--------|----------|----------|------------|
"""
        
        for op_type in sorted(type_stats.keys()):
            stats = type_stats[op_type]
            avg_spd = sum(stats['speedups']) / len(stats['speedups']) if stats['speedups'] else 0
            report += f"| {op_type} | {stats['total']} | {stats['compiled']} ({stats['compiled']/stats['total']*100:.0f}%) | {stats['correct']} ({stats['correct']/stats['total']*100:.0f}%) | {avg_spd:.2f}x |\n"
        
        # 编译失败列表
        if compile_failed:
            report += """
## 编译失败列表

"""
            for level in sorted(set(r.level for r in compile_failed)):
                level_failures = [r for r in compile_failed if r.level == level]
                if level_failures:
                    report += f"""### Level {level}
| Problem ID | 算子名 | 错误信息 |
|------------|--------|----------|
"""
                    for result in sorted(level_failures, key=lambda x: x.problem_id):
                        error = result.generation.error_message[:80] if result.generation and result.generation.error_message else (result.verification.error_message[:80] if result.verification else "未知错误")
                        report += f"| {result.problem_id} | {result.op_name} | {error}... |\n"
                    report += "\n"
        
        # 数值验证失败列表
        if verify_failed:
            report += """## 数值验证失败列表

"""
            for level in sorted(set(r.level for r in verify_failed)):
                level_failures = [r for r in verify_failed if r.level == level]
                if level_failures:
                    report += f"""### Level {level}
| Problem ID | 算子名 | Max Diff | 错误类型 |
|------------|--------|----------|----------|
"""
                    for result in sorted(level_failures, key=lambda x: x.problem_id):
                        max_diff = result.verification.max_diff if result.verification and result.verification.max_diff else "N/A"
                        error_type = "精度超标" if max_diff != "N/A" else "数值异常"
                        report += f"| {result.problem_id} | {result.op_name} | {max_diff} | {error_type} |\n"
                    report += "\n"
        
        # 性能劣化列表
        if perf_degraded:
            report += """## 性能劣化列表（相比 PyTorch）

"""
            for level in sorted(set(r.level for r in perf_degraded)):
                level_degraded = [r for r in perf_degraded if r.level == level]
                if level_degraded:
                    report += f"""### Level {level}
| Problem ID | 算子名 | 加速比 | 劣化倍数 |
|------------|--------|--------|----------|
"""
                    for result in sorted(level_degraded, key=lambda x: x.problem_id):
                        speedup = result.performance.speedup_vs_torch if result.performance else 0
                        degradation = 1.0 / speedup if speedup > 0 else float('inf')
                        report += f"| {result.problem_id} | {result.op_name} | {speedup:.2f}x | {degradation:.2f}x |\n"
                    report += "\n"
        
        report += """
## 详细结果表
| Problem | 算子名 | 类型 | 编译 | 正确 | 加速比 | 状态 |
|---------|--------|------|------|------|--------|------|
"""
        
        for result in sorted(results, key=lambda x: (x.level, x.problem_id)):
            compiled = "✓" if result.verification and result.verification.compiled else "✗"
            correct = "✓" if result.verification and result.verification.correctness else "✗"
            speedup = f"{result.performance.speedup_vs_torch:.2f}x" if result.performance and result.performance.success else "-"
            status = "成功" if result.verification and result.verification.correctness else "失败"
            report += f"| {result.problem_id} | {result.op_name} | {result.op_type} | {compiled} | {correct} | {speedup} | {status} |\n"
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Agent 报告已生成: {output_file}")


class BenchmarkEvaluator:
    """Benchmark 评测器主类"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.timestamp}"
        self.output_dir = os.path.join(
            config.output_root, 
            self.run_id,
            f"agent_{config.agent_name}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化状态管理器
        self.state_manager = StateManager(self.output_dir)
        
        # 记录所有结果
        self.results: List[EvaluationResult] = []
        
        # 加载 agent 配置
        self.agent_config = AgentLoader.load_agent_config(
            config.agent_workspace,
            config.agent_name
        )
    
    def run(self):
        """执行评测"""
        logger.info("=" * 60)
        logger.info("开始 Benchmark 评测")
        logger.info(f"Agent: {self.config.agent_name}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)
        
        # 1. 扫描任务
        logger.info("步骤 1: 扫描任务...")
        tasks = TaskScanner.scan_tasks(
            self.config.benchmark_path,
            self.config.level_problems
        )
        logger.info(f"发现 {len(tasks)} 个任务")
        
        # 2. 过滤已完成的任务
        pending_tasks = []
        for task in tasks:
            if self.config.resume and self.state_manager.is_completed(
                task['level'], task['problem_id']
            ):
                logger.debug(f"跳过已完成任务: Level {task['level']} Problem {task['problem_id']}")
                continue
            pending_tasks.append(task)
        
        logger.info(f"需要评测 {len(pending_tasks)} 个任务 (已完成 {len(tasks) - len(pending_tasks)} 个)")
        
        # 3. 串行执行评测
        logger.info("步骤 2: 开始评测（串行执行）...")
        
        for i, task in enumerate(pending_tasks):
            logger.info(f"进度: {i+1}/{len(pending_tasks)} - Level {task['level']} Problem {task['problem_id']}")
            
            try:
                result = evaluate_single_task(
                    task,
                    self.agent_config,
                    self.config,
                    self.output_dir
                )
                
                # 保存结果
                self.results.append(result)
                
                # 标记完成
                self.state_manager.mark_completed(
                    result.level,
                    result.problem_id
                )
                
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                logger.error(traceback.format_exc())
        
        # 4. 生成报告（始终生成）
        logger.info("步骤 3: 生成报告...")
        
        report_file = os.path.join(self.output_dir, "agent_report.md")
        ReportGenerator.generate_agent_report(
            self.config.agent_name,
            self.results,
            report_file
        )
        
        # 保存原始数据
        all_results_json = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_json, 'w') as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2,
                default=str
            )
        
        logger.info("=" * 60)
        logger.info("评测完成!")
        logger.info(f"结果保存在: {self.output_dir}")
        logger.info(f"报告文件: {report_file}")
        logger.info("=" * 60)


def main():
    """主函数"""
    # 示例配置 - 使用 agent workspace 内的 KernelBench
    # 直接调用 kernelgen-workflow subagent
    agent_workspace = "/mnt/w00934874/agent/code/AscendOpGenAgent/akg-triton/.opencode"
    
    config = TaskConfig(
        agent_name="kernelgen-workflow",  # 直接指定 subagent
        agent_workspace=agent_workspace,
        level_problems={1: [3]},  # Level 1 的 problem 3
        benchmark_path=os.path.join(agent_workspace, "KernelBench", "KernelBench"),
        arch="ascend910b1"
    )
    
    evaluator = BenchmarkEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
