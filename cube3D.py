import tifffile as tiff
from mayavi import mlab
import numpy as np

# 读取图像
image = tiff.imread(r"D:\neoslicegan\2phase_2.tif")
data = image[:150, :150, :150]
print()
data[data == 255] = 250
# data[data == 0] = 180
# data[data == 127] = 0

mlab.figure(bgcolor=(1, 1, 1))
mlab.contour3d(data, contours=[200], opacity=0.3)

@mlab.animate(delay=50)
def animate():
    for i in range(360):
        mlab.view(azimuth=i, elevation=60, distance=500)
        yield

# 启动动画
animate()

# 保存动画为视频文件
# mlab.movie('animation.mp4')

# 显示图像
mlab.show()
