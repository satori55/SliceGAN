import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import einops

# 读取图像
# for i in range(5, 6):

path = fr"C:\Users\SATORI\Downloads\7f_stack_20l_scale2_192.tif"
image = tiff.imread(path)
# image = image[::2, ::2, ::2]
# 提取数据
data = image[20:170, 20:170, 20:170]
# tiff.imwrite("merge.tif", data)
# data = einops.rearrange(data, "x y z -> y x z")
data = data[:, ::-1, :]
print(np.unique(data))
data[data == 255] = 220

# 设置前景和颜色
foreground = data >= 0
colors = np.zeros(foreground.shape + (4,), dtype=float)

# 设置颜色和alpha通道
colors[data <= 0] = [0.91, 0.91, 0.91, 1.0]  # 灰色，不透明
# colors[data > 200] = [241/255, 222/255, 187/255, 1.0]  # yellow，不透明
colors[(data > 10)] = [202/255, 212/255, 231/255, 1.0]  # 青色，不透明

# foreground = (data < 1)
# colors = np.zeros(foreground.shape + (4,), dtype=float)
# colors[foreground] = [round(179/255, 2), round(240/255, 2), round(200/255, 2), 1]

# 画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(foreground, facecolors=colors, edgecolors=None)
ax.axis('off')
# ax.view_init(elev=0, azim=90)
# 保存图像
plt.savefig(f"temp.png", dpi=1000)
# plt.show()
plt.close()
print("finish")
