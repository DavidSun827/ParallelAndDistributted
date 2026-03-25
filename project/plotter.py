import matplotlib.pyplot as plt

# -------------------------
# Data
# -------------------------
P = [1, 2, 4, 8]

runtime_4096 = [0.2051, 0.1771, 0.1653, 0.1880]
runtime_8192 = [0.7677, 0.7134, 0.6399, 0.6717]

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8, 5))

plt.plot(P, runtime_4096, marker='o', label="4096 x 4096")
plt.plot(P, runtime_8192, marker='o', label="8192 x 8192")

plt.xlabel("Number of MPI Processes (P)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Scaling of Our MPI+Numba FFT2 Implementation")
plt.xticks(P)
plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()
plt.tight_layout()

plt.savefig("fft_scaling_runtime.png", dpi=200)
plt.show()