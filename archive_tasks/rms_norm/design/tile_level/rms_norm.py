"""Unified TileLang design for RMSNorm.

This file keeps three specialized prim_funcs:
- merge_n: preferred when N <= 1024
- single_row: preferred when 1024 < N <= 8192
- splitd: preferred when N > 8192

The kernel uses one external dtype for input/output tensors and accumulates in
float32 internally, matching the AscendC interface.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def rms_norm(M, N, eps=1e-5, dtype="float32"):
    block_M = 64
    block_N = 1024
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_M = block_M // vec_num

    row_factor = 8
    row_loops = T.ceildiv(sub_block_M, row_factor)
    n_num = T.ceildiv(N, block_N)
    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    eps_const = T.float32(eps)
    inv_n_const = T.float32(1.0 / N)

    @T.prim_func
    def merge_n(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
        InvRMS: T.Tensor((M,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            gamma_in_ub = T.alloc_ub((1, N), dtype)
            x_in_rows_ub = T.alloc_ub((row_factor, N), dtype)
            out_cast_rows_ub = T.alloc_ub((row_factor, N), dtype)
            inv_rms_cast_ub = T.alloc_ub((row_factor, 1), dtype)
            single_x_in_row_ub = T.alloc_ub((1, N), dtype)
            single_out_cast_row_ub = T.alloc_ub((1, N), dtype)
            single_inv_rms_cast_ub = T.alloc_ub((1, 1), dtype)
            x_ub = T.alloc_ub((row_factor, N), "float32")
            x_sq_ub = T.alloc_ub((row_factor, N), "float32")
            gamma_ub = T.alloc_ub((1, N), "float32")
            gamma_broad_ub = T.alloc_ub((row_factor, N), "float32")
            sum_sq_ub = T.alloc_ub((row_factor, 1), "float32")
            inv_rms_ub = T.alloc_ub((row_factor, 1), "float32")
            rstd_broad_ub = T.alloc_ub((row_factor, N), "float32")
            out_ub = T.alloc_ub((row_factor, N), "float32")

            single_x_ub = T.alloc_ub((1, N), "float32")
            single_x_sq_ub = T.alloc_ub((1, N), "float32")
            single_inv_rms_ub = T.alloc_ub((1, 1), "float32")
            single_out_ub = T.alloc_ub((1, N), "float32")

            inv_n_ub = T.alloc_ub((row_factor, 1), "float32")
            eps_ub = T.alloc_ub((row_factor, 1), "float32")

            reduce_tmp = T.alloc_ub((2 * row_factor * N,), "uint8")
            gamma_bcast_tmp = T.alloc_ub((2 * row_factor, N), "uint8")
            rstd_bcast_tmp = T.alloc_ub((2 * row_factor, N), "uint8")
            single_reduce_tmp = T.alloc_ub((2 * N,), "uint8")

            with T.Scope("V"):
                if need_cast:
                    T.copy(Gamma[0], gamma_in_ub)
                    T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=N)
                else:
                    T.copy(Gamma[0], gamma_ub)
                T.tile.broadcast(gamma_broad_ub, gamma_ub, gamma_bcast_tmp)
                T.tile.fill(inv_n_ub, inv_n_const)
                T.tile.fill(eps_ub, eps_const)

                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for r in T.serial(row_loops):
                            row_base = bx * block_M + vid * sub_block_M + r * row_factor
                            if row_base + row_factor <= M:
                                if need_cast:
                                    T.copy(X[row_base:row_base + row_factor, :], x_in_rows_ub)
                                    T.tile.cast(x_ub, x_in_rows_ub, mode="CAST_NONE", count=row_factor * N)
                                else:
                                    T.copy(X[row_base:row_base + row_factor, :], x_ub)
                                T.tile.mul(x_sq_ub, x_ub, x_ub)
                                T.reduce_sum(x_sq_ub, sum_sq_ub, reduce_tmp, dim=-1)
                                T.tile.mul(sum_sq_ub, sum_sq_ub, inv_n_ub)
                                T.tile.add(sum_sq_ub, sum_sq_ub, eps_ub)
                                T.tile.rsqrt(inv_rms_ub, sum_sq_ub)
                                if need_cast:
                                    T.tile.cast(inv_rms_cast_ub, inv_rms_ub, mode=out_cast_mode, count=row_factor)
                                    T.copy(inv_rms_cast_ub[:, 0], InvRMS[row_base:row_base + row_factor])
                                else:
                                    T.copy(inv_rms_ub[:, 0], InvRMS[row_base:row_base + row_factor])
                                T.tile.broadcast(rstd_broad_ub, inv_rms_ub, rstd_bcast_tmp)
                                T.tile.mul(out_ub, x_ub, rstd_broad_ub)
                                T.tile.mul(out_ub, out_ub, gamma_broad_ub)
                                if need_cast:
                                    T.tile.cast(out_cast_rows_ub, out_ub, mode=out_cast_mode, count=row_factor * N)
                                    T.copy(out_cast_rows_ub, Y[row_base:row_base + row_factor, :])
                                else:
                                    T.copy(out_ub, Y[row_base:row_base + row_factor, :])
                            else:
                                for rr in T.serial(row_factor):
                                    row_idx = row_base + rr
                                    if row_idx < M:
                                        if need_cast:
                                            T.copy(X[row_idx, :], single_x_in_row_ub)
                                            T.tile.cast(single_x_ub, single_x_in_row_ub, mode="CAST_NONE", count=N)
                                        else:
                                            T.copy(X[row_idx, :], single_x_ub)
                                        T.tile.mul(single_x_sq_ub, single_x_ub, single_x_ub)
                                        T.reduce_sum(single_x_sq_ub, single_x_sq_ub[:, 0], single_reduce_tmp, dim=-1)
                                        single_sum_sq = single_x_sq_ub[0, 0] * inv_n_const + eps_const
                                        single_x_sq_ub[0, 0] = single_sum_sq
                                        T.tile.rsqrt(single_inv_rms_ub[:, 0], single_x_sq_ub[:, 0])
                                        single_rstd = single_inv_rms_ub[0, 0]
                                        T.tile.mul(single_out_ub, single_x_ub, single_rstd)
                                        T.tile.mul(single_out_ub, single_out_ub, gamma_ub)
                                        if need_cast:
                                            T.tile.cast(single_inv_rms_cast_ub, single_inv_rms_ub, mode=out_cast_mode, count=1)
                                            T.copy(single_inv_rms_cast_ub[:, 0], InvRMS[row_idx:row_idx + 1])
                                        else:
                                            T.copy(single_inv_rms_ub[:, 0], InvRMS[row_idx:row_idx + 1])
                                        if need_cast:
                                            T.tile.cast(single_out_cast_row_ub, single_out_ub, mode=out_cast_mode, count=N)
                                            T.copy(single_out_cast_row_ub, Y[row_idx, :])
                                        else:
                                            T.copy(single_out_ub, Y[row_idx, :])

    @T.prim_func
    def single_row(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
        InvRMS: T.Tensor((M,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            gamma_in_row_ub = T.alloc_ub((1, N), dtype)
            x_in_row_ub = T.alloc_ub((1, N), dtype)
            out_cast_row_ub = T.alloc_ub((1, N), dtype)
            inv_rms_cast_ub = T.alloc_ub((1, 1), dtype)
            x_ub = T.alloc_ub((1, N), "float32")
            x_sq_ub = T.alloc_ub((1, N), "float32")
            gamma_ub = T.alloc_ub((1, N), "float32")
            inv_rms_ub = T.alloc_ub((1, 1), "float32")
            out_ub = T.alloc_ub((1, N), "float32")

            reduce_tmp = T.alloc_ub((2 * N,), "uint8")

            with T.Scope("V"):
                if need_cast:
                    T.copy(Gamma[0], gamma_in_row_ub)
                    T.tile.cast(gamma_ub, gamma_in_row_ub, mode="CAST_NONE", count=N)
                else:
                    T.copy(Gamma[0], gamma_ub)

                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                if need_cast:
                                    T.copy(X[row_idx, :], x_in_row_ub)
                                    T.tile.cast(x_ub, x_in_row_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(X[row_idx, :], x_ub)
                                T.tile.mul(x_sq_ub, x_ub, x_ub)
                                T.reduce_sum(x_sq_ub, x_sq_ub[:, 0], reduce_tmp, dim=-1)
                                sum_sq = x_sq_ub[0, 0] * inv_n_const + eps_const
                                x_sq_ub[0, 0] = sum_sq
                                T.tile.rsqrt(inv_rms_ub[:, 0], x_sq_ub[:, 0])
                                inv_rms = inv_rms_ub[0, 0]
                                T.tile.mul(out_ub, x_ub, inv_rms)
                                T.tile.mul(out_ub, out_ub, gamma_ub)
                                if need_cast:
                                    T.tile.cast(inv_rms_cast_ub, inv_rms_ub, mode=out_cast_mode, count=1)
                                    T.copy(inv_rms_cast_ub[:, 0], InvRMS[row_idx:row_idx + 1])
                                else:
                                    T.copy(inv_rms_ub[:, 0], InvRMS[row_idx:row_idx + 1])
                                if need_cast:
                                    T.tile.cast(out_cast_row_ub, out_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_row_ub, Y[row_idx, :])
                                else:
                                    T.copy(out_ub, Y[row_idx, :])

    @T.prim_func
    def splitd(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
        InvRMS: T.Tensor((M,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_in_ub = T.alloc_ub((1, block_N), dtype)
            gamma_in_ub = T.alloc_ub((1, block_N), dtype)
            x_ub = T.alloc_ub((1, block_N), "float32")
            x_sq_ub = T.alloc_ub((1, block_N), "float32")
            gamma_ub = T.alloc_ub((1, block_N), "float32")
            inv_rms_ub = T.alloc_ub((1, 1), "float32")
            inv_rms_cast_ub = T.alloc_ub((1, 1), dtype)
            out_ub = T.alloc_ub((1, block_N), "float32")
            out_cast_ub = T.alloc_ub((1, block_N), dtype)

            reduce_tmp = T.alloc_ub((2 * block_N,), "uint8")

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                T.tile.fill(out_ub, T.float32(0))
                                for by in T.serial(n_num):
                                    col_base = by * block_N
                                    valid_n = T.if_then_else(col_base + block_N <= N, block_N, N - col_base)
                                    if need_cast:
                                        T.copy(X[row_idx:row_idx + 1, col_base:col_base + valid_n], x_in_ub[:, 0:valid_n])
                                        T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=valid_n)
                                    else:
                                        T.copy(X[row_idx:row_idx + 1, col_base:col_base + valid_n], x_ub[:, 0:valid_n])

                                    T.tile.mul(x_sq_ub[:, 0:valid_n], x_ub[:, 0:valid_n], x_ub[:, 0:valid_n])
                                    T.tile.add(out_ub[:, 0:valid_n], out_ub[:, 0:valid_n], x_sq_ub[:, 0:valid_n])

                                T.reduce_sum(out_ub, out_ub[:, 0], reduce_tmp, dim=-1)
                                x_sq_ub[0, 0] = out_ub[0, 0] * inv_n_const + eps_const
                                T.tile.rsqrt(inv_rms_ub[:, 0], x_sq_ub[:, 0])
                                inv_rms = inv_rms_ub[0, 0]
                                if need_cast:
                                    T.tile.cast(inv_rms_cast_ub, inv_rms_ub, mode=out_cast_mode, count=1)
                                    T.copy(inv_rms_cast_ub[:, 0], InvRMS[row_idx:row_idx + 1])
                                else:
                                    T.copy(inv_rms_ub[:, 0], InvRMS[row_idx:row_idx + 1])

                                for by in T.serial(n_num):
                                    col_base = by * block_N
                                    valid_n = T.if_then_else(col_base + block_N <= N, block_N, N - col_base)
                                    if need_cast:
                                        T.copy(X[row_idx:row_idx + 1, col_base:col_base + valid_n], x_in_ub[:, 0:valid_n])
                                        T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=valid_n)
                                    else:
                                        T.copy(X[row_idx:row_idx + 1, col_base:col_base + valid_n], x_ub[:, 0:valid_n])

                                    if need_cast:
                                        T.copy(Gamma[col_base:col_base + valid_n], gamma_in_ub[0, 0:valid_n])
                                        T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=valid_n)
                                    else:
                                        T.copy(Gamma[col_base:col_base + valid_n], gamma_ub[0, 0:valid_n])

                                    T.tile.mul(out_ub[:, 0:valid_n], x_ub[:, 0:valid_n], inv_rms)
                                    T.tile.mul(out_ub[:, 0:valid_n], out_ub[:, 0:valid_n], gamma_ub[:, 0:valid_n])

                                    if need_cast:
                                        T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=valid_n)
                                        T.copy(out_cast_ub[:, 0:valid_n], Y[row_idx:row_idx + 1, col_base:col_base + valid_n])
                                    else:
                                        T.copy(out_ub[:, 0:valid_n], Y[row_idx:row_idx + 1, col_base:col_base + valid_n])

    if N <= 1024:
        return merge_n
    if N > 8192:
        return splitd
    return single_row
