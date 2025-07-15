import matplotlib.pyplot as plt
import numpy as np

# 生成0到1的线性空间
values = np.linspace(0, 1, 256)

# 取viridis colormap
cmap = plt.cm.viridis

# 将数值映射为RGBA颜色
colors = cmap(values)

# 显示颜色条
plt.figure(figsize=(8, 2))
plt.title('Matplotlib Viridis Colormap')
plt.imshow([colors], aspect='auto')
plt.axis('off')
plt.show()
