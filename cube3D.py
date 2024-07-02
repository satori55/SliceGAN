import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个 100x100x100 的 3D 矩阵，包含 3 种不同的值
np.random.seed(0)
matrix = np.random.choice([0, 1, 2], size=(100, 100, 100))

# 创建一个颜色映射
colors = np.empty(matrix.shape, dtype=object)
colors[matrix == 0] = 'r'
colors[matrix == 1] = 'g'
colors[matrix == 2] = 'b'

# 设置旋转的角度
angles = np.linspace(0, 360, 36)

# 创建一个 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 矩阵中的每个点
def plot_matrix(ax, matrix, colors):
    ax.voxels(matrix == 0, facecolors='r', edgecolor='k', alpha=0.5)
    ax.voxels(matrix == 1, facecolors='g', edgecolor='k', alpha=0.5)
    ax.voxels(matrix == 2, facecolors='b', edgecolor='k', alpha=0.5)

# 保存每个角度的图像
for angle in angles:
    ax.clear()
    plot_matrix(ax, matrix, colors)
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.1)
    break

plt.show()
