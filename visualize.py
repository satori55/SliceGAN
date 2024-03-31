import tifffile as tiff
from mayavi import mlab
import numpy as np


image = tiff.imread('2phase.tif')
data = image[:50, :50, :50]

# visualize the 3D image
src = mlab.pipeline.scalar_field(data)
mlab.pipeline.iso_surface(src, contours=[1], opacity=0.15) # 0.5
mlab.show()
