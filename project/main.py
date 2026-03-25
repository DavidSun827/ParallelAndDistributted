from mpi4py import MPI
import numpy as np
import argparse
import os
'''
this is the first push updata to gihub

'''
try:
    import pyfftw

    HAVE_PYFFTW = True
    # 打开接口缓存，便于重复调用
    pyfftw.interfaces.cache.enable()
except Exception as e:
    HAVE_PYFFTW = False
    _PYFFTW_IMPORT_ERROR = str(e)

from numba import njit, prange
import math


# ---------- 基础工具 ----------
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------- Numba 优化的 Cooley–Tukey 1D FFT ----------
@njit(fastmath=True, cache=True)
def _bit_reverse_int(i: int, m_bits: int) -> int:
    r = 0
    x = i
    for _ in range(m_bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


@njit(fastmath=True, cache=True)
def _bit_reverse_permute_inplace(a: np.ndarray) -> None:
    n = a.shape[0]
    # 假设 n 是 2 的幂
    m_bits = int(math.log2(n))
    for i in range(n):
        j = _bit_reverse_int(i, m_bits)
        if j > i:
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp


@njit(fastmath=True, cache=True)
def ct_fft_radix2_inplace(a: np.ndarray) -> None:
    """
    就地 1D radix-2 FFT：a 为 complex128 向量（长度为 2 的幂）。
    采用 stage-by-stage 蝶形，每个 stage 对 block 并行。
    """
    n = a.shape[0]
    _bit_reverse_permute_inplace(a)

    s = 2
    while s <= n:
        half = s // 2
        angle = -2.0 * math.pi / s
        wm = complex(math.cos(angle), math.sin(angle))

        blocks = n // s
        for b in range(blocks):
            base = b * s
            # 蝶形：逐列推进旋转因子
            w = 1.0 + 0.0j
            for j in range(half):
                u = a[base + j]
                v = a[base + j + half]
                t = w * v
                a[base + j] = u + t
                a[base + j + half] = u - t
                w *= wm
        s <<= 1


@njit(parallel=True, fastmath=True, cache=True)
def fft_rows_inplace(mat: np.ndarray) -> None:
    """对 mat 的每一行做 1D FFT（行内并行）。"""
    nrows = mat.shape[0]
    for r in prange(nrows):
        row = mat[r, :]
        ct_fft_radix2_inplace(row)


@njit(parallel=True, fastmath=True, cache=True)
def fft_cols_inplace(mat: np.ndarray) -> None:
    """
    对 mat 的每一列做 1D FFT（列间并行）。
    为保证连续内存，对每列 copy 出来做 FFT，再写回。
    """
    nrows, ncols = mat.shape
    for c in prange(ncols):
        col = np.empty(nrows, dtype=np.complex128)
        # 提取列
        for i in range(nrows):
            col[i] = mat[i, c]
        # FFT
        ct_fft_radix2_inplace(col)
        # 写回
        for i in range(nrows):
            mat[i, c] = col[i]


@njit(cache=True, fastmath=True, parallel=True)
def _jit_warmup() -> None:
    """极小尺寸的预热，触发 JIT 编译。"""
    v = np.zeros(8, dtype=np.complex128)
    ct_fft_radix2_inplace(v)
    M = np.zeros((8, 16), dtype=np.complex128)
    fft_rows_inplace(M)
    fft_cols_inplace(M)


def fftw_fft2_time(
        x: np.ndarray,
        threads: int | None = None,
        effort: str = "FFTW_MEASURE",
        warmup: int = 1,
):
    """
    用 pyFFTW 进行 2D FFT 并计时。
    返回: (y, exec_time, plan_time, used_threads)
      - y: FFT 结果
      - exec_time: 单次执行时间（不含计划时间）
      - plan_time: 规划(plan)时间（FFTW_MEASURE 下会较长）
      - used_threads: 实际使用的线程数
    """
    if not HAVE_PYFFTW:
        raise RuntimeError("pyFFTW not available")

    if threads is None:
        threads = int(os.environ.get("FFTW_NUM_THREADS", os.cpu_count() or 1))

    # 对齐的输入/输出缓冲
    in_arr = pyfftw.empty_aligned(x.shape, dtype="complex128")
    out_arr = pyfftw.empty_aligned(x.shape, dtype="complex128")
    in_arr[:] = x

    # 规划
    t_plan0 = MPI.Wtime()
    plan = pyfftw.FFTW(
        in_arr,
        out_arr,
        axes=(0, 1),
        direction="FFTW_FORWARD",
        threads=threads,
        flags=(effort,),
    )
    t_plan1 = MPI.Wtime()

    # 预热（不计时）
    for _ in range(warmup):
        plan()

    # 单次执行计时
    t0 = MPI.Wtime()
    plan()
    t1 = MPI.Wtime()

    return out_arr.copy(), (t1 - t0), (t_plan1 - t_plan0), threads


# ---------- 并行 2D FFT ----------
def parallel_fft2_ct(x_global: np.ndarray | None, comm: MPI.Comm, root: int = 0):
    """
    输入:
        x_global: 仅 root 提供的 (N1, N2) 复数组，其它进程传 None
    返回:
        (X2, comp_time, comm_time)
        - X2: 仅 root 返回 (N1, N2) 的 2D FFT 结果，其它进程返回 None
        - comp_time: 各 rank 中最大计算时间 (rows+cols)
        - comm_time: 各 rank 中最大通信时间 (两次 Alltoall 的和)

    约束:
        N1 % P == 0, N2 % P == 0, 且 N1、N2、P 均为 2 的幂
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        N1, N2 = map(int, x_global.shape)
    else:
        N1 = N2 = 0
    N1 = comm.bcast(N1, root=root)
    N2 = comm.bcast(N2, root=root)

    assert N1 % size == 0 and N2 % size == 0, "N1 and N2 must be divisible by #procs"
    assert is_power_of_two(N1) and is_power_of_two(N2), "N1, N2 must be powers of two"
    assert is_power_of_two(size), "P (world size) must be power of two"

    n1 = N1 // size  # 每进程行数（行切片）
    c2 = N2 // size  # 列切片宽度（列切片阶段每个进程持有的列数）

    # Step 0: Scatter 行切片 (row-slab)
    if rank == root:
        send_rows = x_global.astype(np.complex128, copy=False)
    else:
        send_rows = None

    local_rows = np.empty((n1, N2), dtype=np.complex128)
    comm.Scatter(send_rows, local_rows, root=root)

    # ===== 计时变量（局部）=====
    comp_rows = 0.0
    comp_cols = 0.0
    comm_alltoall1 = 0.0
    comm_alltoall2 = 0.0

    # Step 1: 行内 FFT（Numba 并行）
    t_comp_rows0 = MPI.Wtime()
    fft_rows_inplace(local_rows)
    t_comp_rows1 = MPI.Wtime()
    comp_rows = t_comp_rows1 - t_comp_rows0

    # Step 2: 行切片 -> 列切片 的 Alltoall 重分布
    sendbuf = np.empty((size, n1, c2), dtype=np.complex128)
    for d in range(size):
        sendbuf[d, :, :] = local_rows[:, d * c2: (d + 1) * c2]
    recvbuf = np.empty_like(sendbuf)

    t_comm1_0 = MPI.Wtime()
    comm.Alltoall(sendbuf, recvbuf)
    t_comm1_1 = MPI.Wtime()
    comm_alltoall1 = t_comm1_1 - t_comm1_0

    # 拼接为列切片布局 (N1, c2)
    col_slab = np.vstack([recvbuf[s, :, :] for s in range(size)])  # 形状 (N1, c2)

    # Step 3: 列向 FFT（Numba 并行）
    t_comp_cols0 = MPI.Wtime()
    fft_cols_inplace(col_slab)
    t_comp_cols1 = MPI.Wtime()
    comp_cols = t_comp_cols1 - t_comp_cols0

    # Step 4: 列切片 -> 行切片 的反向 Alltoall
    sendbuf2 = np.empty((size, n1, c2), dtype=np.complex128)
    for d in range(size):
        sendbuf2[d, :, :] = col_slab[d * n1: (d + 1) * n1, :]
    recvbuf2 = np.empty_like(sendbuf2)

    t_comm2_0 = MPI.Wtime()
    comm.Alltoall(sendbuf2, recvbuf2)
    t_comm2_1 = MPI.Wtime()
    comm_alltoall2 = t_comm2_1 - t_comm2_0

    # 重建本地行切片 (n1, N2)
    local_out = np.empty((n1, N2), dtype=np.complex128)
    for s in range(size):
        local_out[:, s * c2: (s + 1) * c2] = recvbuf2[s, :, :]

    # Step 5: Gather 回 root
    if rank == root:
        X2 = np.empty((N1, N2), dtype=np.complex128)
    else:
        X2 = None
    comm.Gather(local_out, X2, root=root)

    # ===== 全局规约：把每个阶段的时间取 max（HPC 常用做法）=====
    comp_rows_max = comm.allreduce(comp_rows, op=MPI.MAX)
    comp_cols_max = comm.allreduce(comp_cols, op=MPI.MAX)
    comm_alltoall1_max = comm.allreduce(comm_alltoall1, op=MPI.MAX)
    comm_alltoall2_max = comm.allreduce(comm_alltoall2, op=MPI.MAX)

    comp_time = comp_rows_max + comp_cols_max
    comm_time = comm_alltoall1_max + comm_alltoall2_max

    return X2, comp_time, comm_time


# ---------- 主程序 ----------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        description="Parallel 2D Cooley–Tukey FFT with mpi4py + Numba"
    )
    parser.add_argument(
        "--N1", type=int, default=8192, help="rows (power of two, divisible by P)"
    )
    parser.add_argument(
        "--N2", type=int, default=8192, help="cols (power of two, divisible by P)"
    )
    parser.add_argument(
        "--check", action="store_true", help="compare with numpy.fft.fft2 on root"
    )
    args = parser.parse_args()

    # 预热（所有进程都做，避免把 JIT 时间算进性能）
    _jit_warmup()

    N1, N2 = args.N1, args.N2
    x_global = None
    if rank == 0:
        rng = np.random.default_rng(123)
        x_global = (
            rng.standard_normal((N1, N2)) + 1j * rng.standard_normal((N1, N2))
        ).astype(np.complex128)

    # ===== 计时：并行 FFT2 =====
    comm.Barrier()
    t0 = MPI.Wtime()
    X2, comp_time, comm_time = parallel_fft2_ct(x_global, comm, root=0)
    comm.Barrier()
    t1 = MPI.Wtime()
    par_time = t1 - t0

    # ===== 计时：NumPy FFT2（仅 root）=====
    comm.Barrier()
    if rank == 0:
        t2 = MPI.Wtime()
        ref = np.fft.fft2(x_global)  # 也用于正确性对比
        t3 = MPI.Wtime()
        np_time = t3 - t2
    else:
        ref = None
        np_time = None

    # ===== 计时：FFTW（仅 root，若可用）=====
    if rank == 0 and HAVE_PYFFTW:
        try:
            ref_fftw, fftw_time, fftw_plan_time, fftw_threads = fftw_fft2_time(
                x_global,
                threads=int(os.environ.get("FFTW_NUM_THREADS", os.cpu_count() or 1)),
                effort="FFTW_MEASURE",  # 如需更快计划可改 'FFTW_ESTIMATE'
                warmup=1,
            )
        except Exception as e:
            print(f"[Perf] FFTW timing failed: {e}")
            ref_fftw, fftw_time, fftw_plan_time, fftw_threads = (
                None,
                None,
                None,
                None,
            )
    else:
        ref_fftw, fftw_time, fftw_plan_time, fftw_threads = (
            None,
            None,
            None,
            None,
        )

    # 广播单值，便于在非 root 进程可用（可选）
    np_time = comm.bcast(np_time, root=0)
    fftw_time = comm.bcast(fftw_time, root=0)

    if rank == 0:
        print(
            f"[Perf] parallel_fft2_ct wall time: {par_time:.6f} s for ({N1}x{N2}), P={size}"
        )
        print(f"[Perf] numpy.fft.fft2 wall time (root only): {np_time:.6f} s")
        if fftw_time is not None:
            print(
                f"[Perf] FFTW exec time (root only, threads={fftw_threads}): "
                f"{fftw_time:.6f} s (plan: {fftw_plan_time:.6f} s, excluded from exec time)"
            )

        # 简单 speedup 指标
        if par_time > 0:
            print(f"[Perf] speedup (numpy/parallel): {np_time / par_time:.2f}x")
            if fftw_time is not None:
                print(f"[Perf] speedup (fftw/parallel): {fftw_time / par_time:.2f}x")
                print(f"[Perf] speedup (numpy/fftw): {np_time / fftw_time:.2f}x")

        # 计算 vs 通信 breakdown（两次 Alltoall）
        print(
            f"[Breakdown] compute time (rows+cols, max over ranks): {comp_time:.6f} s"
        )
        print(
            f"[Breakdown] communication time (2x Alltoall, max over ranks): {comm_time:.6f} s"
        )
        if par_time > 0:
            print(
                f"[Breakdown] comm / total parallel time: {comm_time / par_time:.3f}"
            )

        # 正确性检查
        if args.check:
            err = np.max(np.abs(X2 - ref))
            print(f"[Check] max abs error vs numpy.fft.fft2: {err:.3e}")

            if ref_fftw is not None:
                err_fftw = np.max(np.abs(X2 - ref_fftw))
                print(f"[Check] max abs error vs FFTW: {err_fftw:.3e}")

            x_back = np.fft.ifft2(X2)
            err_ifft = np.max(np.abs(x_back - x_global))
            rel_ifft = np.linalg.norm(x_back - x_global) / np.linalg.norm(x_global)
            print(f"[Roundtrip] max abs error ifft2(FFT2(x)) vs x: {err_ifft:.3e}")
            print(f"[Roundtrip] relative L2 error: {rel_ifft:.3e}")

    # 若未安装 pyFFTW，提示一次
    if rank == 0 and not HAVE_PYFFTW:
        print(
            "[Perf] pyFFTW not available; skipping FFTW timing. "
            "Install with: pip install pyfftw"
        )
        print(f"        Import error: {_PYFFTW_IMPORT_ERROR}")


if __name__ == "__main__":
    main()