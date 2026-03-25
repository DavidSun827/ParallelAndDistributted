import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
methods = ["Our MPI FFT2 (P=4)", "NumPy FFT2", "FFTW"]

runtime_4096 = [0.1653, 0.1975, 0.0897]
runtime_8192 = [0.6399, 0.9361, 0.4483]

x = np.arange(len(methods))  # 3 bars
width = 0.35                 # width of the bars

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8,5))

plt.bar(x - width/2, runtime_4096, width, label="4096 × 4096")
plt.bar(x + width/2, runtime_8192, width, label="8192 × 8192")

plt.xticks(x, methods, rotation=15)
plt.ylabel("Runtime (seconds)")
plt.title("Performance Comparison: Our Best MPI FFT2 vs NumPy and FFTW")
plt.legend()

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("fft_best_vs_numpy_fftw.png", dpi=200)
plt.show()