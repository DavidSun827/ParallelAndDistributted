'''
mpiexec -n 8 python main_with_profiler.py --N1 4096 --N2 4096
'''


from mpi4py import MPI
import numpy as np
import argparse
from numba import njit, prange
import math


# ---------- 基础工具 ----------
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------- Numba Cooley–Tukey 1D FFT ----------
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
    m_bits = int(math.log2(n))
    for i in range(n):
        j = _bit_reverse_int(i, m_bits)
        if j > i:
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp


@njit(fastmath=True, cache=True)
def ct_fft_radix2_inplace(a: np.ndarray) -> None:
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
            w = 1.0 + 0.0j
            for j in range(half):
                u = a[base + j]
                v = a[base + j + half]
                t = w * v
                a[base + j] = u + t
                a[base + j + half] = u - t
                w *= wm
        s <<= 1


# ---------- 行 FFT ----------
@njit(parallel=True, fastmath=True, cache=True)
def fft_rows_inplace(mat: np.ndarray) -> None:
    nrows = mat.shape[0]
    for r in prange(nrows):
        ct_fft_radix2_inplace(mat[r])


# ---------- 列 FFT ----------
@njit(parallel=True, fastmath=True, cache=True)
def fft_cols_inplace(mat: np.ndarray) -> None:
    nrows, ncols = mat.shape
    for c in prange(ncols):
        col = np.empty(nrows, dtype=np.complex128)
        for i in range(nrows):
            col[i] = mat[i, c]
        ct_fft_radix2_inplace(col)
        for i in range(nrows):
            mat[i, c] = col[i]


# ---------- JIT 预热 ----------
@njit(cache=True, fastmath=True, parallel=True)
def _jit_warmup():
    v = np.zeros(8, dtype=np.complex128)
    ct_fft_radix2_inplace(v)
    M = np.zeros((8, 16), dtype=np.complex128)
    fft_rows_inplace(M)
    fft_cols_inplace(M)


# ---------- 并行 2D FFT ----------
def parallel_fft2_ct(x_global: np.ndarray | None, comm: MPI.Comm, root=0):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        N1, N2 = x_global.shape
    else:
        N1 = N2 = 0

    N1 = comm.bcast(N1, root=root)
    N2 = comm.bcast(N2, root=root)

    assert N1 % size == 0 and N2 % size == 0
    assert is_power_of_two(N1) and is_power_of_two(N2)

    n1 = N1 // size
    c2 = N2 // size

    # Scatter rows
    if rank == root:
        send_rows = x_global.astype(np.complex128, copy=False)
    else:
        send_rows = None

    local_rows = np.empty((n1, N2), dtype=np.complex128)
    comm.Scatter(send_rows, local_rows, root=root)

    # ----- Step 1: Row FFT -----
    t0 = MPI.Wtime()
    fft_rows_inplace(local_rows)
    comp_rows = MPI.Wtime() - t0

    # ----- Step 2: Alltoall (row→col slab) -----
    sendbuf = np.empty((size, n1, c2), dtype=np.complex128)
    for d in range(size):
        sendbuf[d] = local_rows[:, d*c2:(d+1)*c2]

    recvbuf = np.empty_like(sendbuf)
    t1 = MPI.Wtime()
    comm.Alltoall(sendbuf, recvbuf)
    comm1 = MPI.Wtime() - t1

    col_slab = np.vstack([recvbuf[s] for s in range(size)])

    # ----- Step 3: Column FFT -----
    t2 = MPI.Wtime()
    fft_cols_inplace(col_slab)
    comp_cols = MPI.Wtime() - t2

    # ----- Step 4: Alltoall (col→row slab) -----
    sendbuf2 = np.empty((size, n1, c2), dtype=np.complex128)
    for d in range(size):
        sendbuf2[d] = col_slab[d*n1:(d+1)*n1]

    recvbuf2 = np.empty_like(sendbuf2)
    t3 = MPI.Wtime()
    comm.Alltoall(sendbuf2, recvbuf2)
    comm2 = MPI.Wtime() - t3

    # Reconstruct local output
    local_out = np.empty((n1, N2), dtype=np.complex128)
    for s in range(size):
        local_out[:, s*c2:(s+1)*c2] = recvbuf2[s]

    # Gather output
    if rank == root:
        X2 = np.empty((N1, N2), dtype=np.complex128)
    else:
        X2 = None
    comm.Gather(local_out, X2, root=root)

    comp_total = comm.allreduce(comp_rows + comp_cols, op=MPI.MAX)
    comm_total = comm.allreduce(comm1 + comm2, op=MPI.MAX)

    return X2, comp_total, comm_total


# ---------- 主程序 ----------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument("--N1", type=int, default=4096)
    parser.add_argument("--N2", type=int, default=4096)
    args = parser.parse_args()

    # JIT warmup (important)
    _jit_warmup()

    if rank == 0:
        rng = np.random.default_rng(0)
        x_global = (rng.standard_normal((args.N1, args.N2)) +
                    1j * rng.standard_normal((args.N1, args.N2))).astype(np.complex128)
    else:
        x_global = None

    comm.Barrier()
    t0 = MPI.Wtime()
    _, comp, comm_t = parallel_fft2_ct(x_global, comm, root=0)
    comm.Barrier()
    t1 = MPI.Wtime()

    if rank == 0:
        total = t1 - t0
        print(f"[Perf] parallel time: {total:.6f}s   P={comm.Get_size()}")
        print(f"[Breakdown] compute: {comp:.6f}s")
        print(f"[Breakdown] communication: {comm_t:.6f}s")
        print(f"[Breakdown] comm/total = {comm_t/total:.3f}")


# ---------- Profiler Wrapper (Correct Version) ----------
if __name__ == "__main__":
    import cProfile, pstats
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # First run: trigger Numba JIT (DO NOT PROFILE)
    main()

    # Second run: profiler after JIT is done
    profile_file = f"profile_rank{rank}.prof"
    cProfile.run("main()", profile_file)

    if rank == 0:
        print(f"[Profiler] Profiles saved: profile_rank*.prof")
        p = pstats.Stats(profile_file)
        p.sort_stats("tottime").print_stats(20)