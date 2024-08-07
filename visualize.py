from sympy import N
import tifffile as tiff
from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


image = tiff.imread(r"C:\Users\SATORI\Downloads\3_stack_20_scale2_192(1).tif")
data = image#[:50, :50, :50]

foreground = data > 0
colors = np.zeros(foreground.shape + (4,), dtype=float)
colors[data >200] = [round(254/255, 2), round(238/255, 2), round(218/255, 2), 0.8]
colors[(data > 0) & (data <= 200)] = [round(145/255, 2), round(210/255, 2), round(228/255, 2), 0.8]

# foreground = (data < 1)
# colors = np.zeros(foreground.shape + (4,), dtype=float)
# colors[foreground] = [round(184/255, 2), round(219/255, 2), round(179/255, 2), 1]

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_facecolor("Cyan")
# ax.set_box_aspect([depth, height, width])
ax.voxels(foreground, facecolors=colors, edgecolors=None)
ax.axis('off')
ax.view_init(elev=30, azim=60)  # Set view angle
plt.savefig("3d_plot_3_gt.png", dpi=300, bbox_inches='tight')

plt.close()
print("finish")
