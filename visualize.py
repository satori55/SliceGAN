import tifffile as tiff
from mayavi import mlab
import numpy as np


image = tiff.imread('2phase.tif')
# 生成示例数据，这里是一个3D数组
data = image[:50, :50, :50]
# 创建一个球体
x0, y0, z0, radius = 25, 25, 25, 10
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        for z in range(data.shape[2]):
            if (x - x0)**2 + (y - y0)**2 + (z - z0)**2 < radius**2:
                data[x, y, z] = 1

# 可视化
src = mlab.pipeline.scalar_field(data)
mlab.pipeline.iso_surface(src, contours=[1], opacity=0.15) # 设置透明度为0.5
mlab.show()
