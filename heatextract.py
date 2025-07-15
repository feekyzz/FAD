import numpy as np
import matplotlib.pyplot as plt
import cv2

# 你手动定义的9个3x3卷积核，每个是一个numpy数组
kernels = [
    np.array([
        [0.997, 0.532, 0.626],
        [0.078, 0.774, 0.700],
        [0.000, 0.743, 0.551],
    ], dtype=np.float32),

    np.array([
        [0.043, 0.012, 0.000],
        [0.563, 0.422, 0.457],
        [0.477, 0.997, 0.649],
    ], dtype=np.float32),

    np.array([
        [0.583, 0.477, 0.997],
        [0.759, 0.669, 0.000],
        [0.477, 0.583, 0.000],
    ], dtype=np.float32),

    np.array([
        [0.000, 0.516, 0.981],
        [0.473, 0.035, 0.997],
        [0.246, 0.395, 0.931],
    ], dtype=np.float32),

    np.array([
        [0.821, 0.309, 0.837],
        [0.696, 0.000, 0.731],
        [0.997, 0.895, 0.946],
    ], dtype=np.float32),

    np.array([
        [0.766, 0.583, 0.000],
        [0.997, 0.141, 0.809],
        [0.856, 0.371, 0.414],
    ], dtype=np.float32),

    np.array([
        [0.063, 0.673, 0.418],
        [0.000, 0.997, 0.700],
        [0.805, 0.113, 0.297],
    ], dtype=np.float32),

    np.array([
        [0.606, 0.997, 0.633],
        [0.516, 0.778, 0.469],
        [0.414, 0.000, 0.336],
    ], dtype=np.float32),

    np.array([
        [0.422, 0.759, 0.004],
        [0.884, 0.997, 0.000],
        [0.446, 0.219, 0.825],
    ], dtype=np.float32),
]

kernel_size = 3

# 放大卷积核方便显示
kernels_resized = [cv2.resize(k, (100, 100), interpolation=cv2.INTER_NEAREST) for k in kernels]

# 计算傅里叶频谱（log幅度）
def compute_magnitude_spectrum(kernel):
    padded = np.zeros((128, 128), dtype=np.float32)
    padded[:kernel_size, :kernel_size] = kernel
    f_transform = np.fft.fft2(padded)
    f_shift = np.fft.fftshift(f_transform)
    mag = np.log(np.abs(f_shift) + 1e-6)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag_norm

spectrums = [compute_magnitude_spectrum(k) for k in kernels]

# 画空间域卷积核
# 空间域图像
fig1, axes1 = plt.subplots(3, 3, figsize=(9, 9))
# fig1.suptitle("Spatial Domain: 9 Kernels (viridis)", fontsize=16)
for i, ax in enumerate(axes1.flat):
    im = ax.imshow(kernels_resized[i], cmap='summer')
    # im = ax.imshow(kernels_resized[i], cmap='summer')plasma
    # ax.set_title(f'Kernel {i+1}')
    ax.axis('off')
    # fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 傅里叶频域图像
fig2, axes2 = plt.subplots(3, 3, figsize=(9, 9))
# fig2.suptitle("Frequency Domain: FFT Magnitude (viridis, vmax=90th percentile)", fontsize=16)
for i, ax in enumerate(axes2.flat):
    vmax = np.percentile(spectrums[i], 90)
    im = ax.imshow(spectrums[i], cmap='magma', vmin=0, vmax=vmax)
    # im = ax.imshow(spectrums[i], cmap='viridis', vmin=0, vmax=vmax)
    ax.axis('off')
    # fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.show()
