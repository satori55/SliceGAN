import tifffile as tiff
from mayavi import mlab
import numpy as np

# 读取图像
image = tiff.imread('2phase.tif')
data = image[:50, :50, :50]

# 创建Mayavi中的标量场
src = mlab.pipeline.scalar_field(data)

# 为两个不同的值创建等值面，分别设置不同的颜色
mlab.pipeline.iso_surface(src, contours=[1], color=(1, 0, 0), opacity=0.15)  # 红色
mlab.pipeline.iso_surface(src, contours=[0], color=(0, 1, 0), opacity=0.15)  # 绿色

# 显示图像
mlab.show()
