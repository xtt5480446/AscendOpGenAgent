"""Gate 层包。

- common.py:        所有分支共享的通用层校验（反作弊、文件存在、baseline、eval_status）
- branch_precision: 精度（原 precision_gate.py 精度逻辑抽离）
- branch_build:     编译失败
- branch_import:    import 失败（仅 kernel_side）
- branch_runtime:   运行时段错误
- branch_timeout:   超时
"""
