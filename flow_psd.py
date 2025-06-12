from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from openpyxl import load_workbook
from matplotlib.colors import LinearSegmentedColormap

# Load Excel
data_path = r"C:\Users\SATORI\Downloads\FFc_dataset(1).xlsx"
wb = load_workbook(data_path)
ws = wb["Sheet1"]

psd_data = []
flow_data = []
aspect_data = []

psd_col = 3
flow_col = 8
aspect_col = 11

# 读取数据
for idx_row in range(3, 123):
    psd_data.append(ws.cell(row=idx_row, column=psd_col).value)
    flow_data.append(ws.cell(row=idx_row, column=flow_col).value)
    aspect_data.append(ws.cell(row=idx_row, column=aspect_col).value)

# 转换为 NumPy 数组
x = np.array(psd_data, dtype=float) / 1000  # PSD 数据
y = np.array(aspect_data, dtype=float)  # Aspect 数据
z = np.array(flow_data, dtype=float)  # Flow 数据

# 确认数据长度是否一致
assert len(x) == len(y) == len(z), "数据长度不一致！"

# Normalize z values to map to a colormap
norm = plt.Normalize(z.min(), z.max())
colors = plt.cm.viridis(norm(z))

# Create 3D scatter plot
fig = plt.figure(facecolor='white', figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=colors, marker='o', s=50)

# Add color bar
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
# cbar.set_label('z Value')

# Label axes with updated font properties
ax.set_xlabel('PSD', fontsize=26, fontname="Arial", fontweight='bold', labelpad=20)
ax.set_ylabel('Aspect Ratio', fontsize=26, fontname="Arial", fontweight='bold', labelpad=20)
ax.set_zlabel('Flow', fontsize=26, fontname="Arial", fontweight='bold', rotation=90, labelpad=20)


# Set tick labels font properties
ax.tick_params(axis='both', which='major', labelsize=26, width=6)  # Major ticks width and label size
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
    tick.set_fontweight('bold')
    tick.set_fontsize(40)  # Additional font size control

for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")
    tick.set_fontweight('bold')
    tick.set_fontsize(40)  # Additional font size control
for tick in ax.get_zticklabels():
    tick.set_fontname("Arial")
    tick.set_fontweight('bold')
    tick.set_fontsize(40)  # Additional font size control

# ax.tick_params(axis='x', labelsize=34)  # X-axis tick labels
# ax.tick_params(axis='y', labelsize=34)  # Y-axis tick labels
# ax.tick_params(axis='z', labelsize=34)  # Z-axis tick labels

# Remove gridlines
ax.grid(False)

# Set axis pane colors to white and fully opaque
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove spines to ensure pure white background
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

ax.xaxis.pane.set_linewidth(3)  # Set border line width
ax.yaxis.pane.set_linewidth(3)
ax.zaxis.pane.set_linewidth(3)

# Adjust tick parameters for thickness
ax.tick_params(axis='x', which='major', width=2, labelsize=12)  # Major ticks width and label size
ax.tick_params(axis='y', which='major', width=2, labelsize=12)
ax.tick_params(axis='z', which='major', width=2, labelsize=12)

# Save the plot
plt.savefig("3d_scatter.png", dpi=300, facecolor='white', bbox_inches='tight')

# Show the plot
plt.show()


'''
# 创建网格，用于绘制曲面
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 50),
                     np.linspace(y.min(), y.max(), 50))

# 使用 griddata 插值
zz = griddata((x, y), z, (xx, yy), method='linear')  # 使用线性插值方法

# 掩盖无数据区域
mask = np.isnan(zz)  # 找出插值结果为 NaN 的区域
zz = np.where(mask, np.nan, zz)  # 将 NaN 保留在 zz 中

# 限制插值值范围（裁剪）
z_min, z_max = z.min(), z.max()
zz = np.clip(zz, z_min, z_max)  # 限制 zz 在 z_min 和 z_max 范围内

# 定义浅蓝到粉红渐变色的 colormap
colors = [(0.8, 0.9, 1), (0.9, 0.7, 1), (1, 0.5, 0.8), (1, 0.7, 0.9), (1, 0.8, 0.9)]
cmap = LinearSegmentedColormap.from_list('blue_pink_colormap', colors, N=256)

# 绘制 3D 图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用 zz 值的颜色映射
norm = plt.Normalize(vmin=z_min, vmax=z_max)  # 归一化 z 值范围
colors = cmap(norm(zz))  # 应用自定义渐变色
surf = ax.plot_surface(
    xx, yy, zz, facecolors=colors,
    rstride=1, cstride=1,  # 控制步长
    linewidth=0,  # 设为 0，完全去掉格子线
    antialiased=True,  # 确保表面更加平滑
    alpha=1  # 设置完全不透明
)

# 添加颜色条，并明确指定与颜色映射的关联
m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
m.set_array(z)  # 明确设置颜色条的数据范围
cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)  # 将颜色条与当前轴关联
cbar.set_label('Z Value')  # 颜色条标签

# 设置坐标轴标签
ax.set_xlabel('X Label')  # 设置 X 轴标签
ax.set_ylabel('Y Label')  # 设置 Y 轴标签
ax.set_zlabel('Z Label')  # 设置 Z 轴标签

plt.show()
'''

