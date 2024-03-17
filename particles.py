import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage.measure import label, regionprops


image = tiff.imread('stacked_binary_images.tif')

# step2: label particles
labeled_image, num_features = label(image, return_num=True)

# step3: properties analysis
properties = regionprops(labeled_image)

# equivalent radius
radii = []

for prop in properties:
    # equivalent radius
    volume = prop.area  # volume
    radius = ((3 * volume) / (4 * np.pi)) ** (1/3)
    radii.append(radius)

# mean and std
average_radius = float(np.mean(radii))
std_dev_radius = float(np.std(radii))

# plot
plt.figure(figsize=(10, 6))
plt.hist(radii, bins=30, color='skyblue', edgecolor='black')
plt.title('Particle Radius Distribution')
plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.grid(True)

# mean std
plt.axvline(average_radius, color='r', linestyle='dashed', linewidth=1)
plt.axvline(average_radius - std_dev_radius, color='g', linestyle='dashed', linewidth=1)
plt.axvline(average_radius + std_dev_radius, color='g', linestyle='dashed', linewidth=1)
plt.legend(['Mean Radius', 'Std Deviation'])

plt.show()

print(average_radius)
print(std_dev_radius)
