import cv2
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage.measure import label, regionprops
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr


images = tiff.imread('5slices_RMS_3.tif')

# equivalent radius
radii = []

for i in range(images.shape[0]):
    # step2: label particles
    labeled_image, num_features = label(images[i], return_num=True)

    # step3: properties analysis
    properties = regionprops(labeled_image)

    for prop in properties:
        # equivalent radius
        s = prop.area  # volume
        radius = (s / np.pi) ** (1/2)
        radii.append(radius)

# mean and std
average_radius = float(np.mean(radii))
std_dev_radius = float(np.std(radii))
# print(radii)
bins = np.arange(0, 40.2, 0.2)
hist, bin_edges = np.histogram(radii, bins=bins)

# 打印每个分组的边界和计数
# for i in range(len(hist)):
#     print(f"分组 {i+1}: 边界 = ({bin_edges[i]}, {bin_edges[i+1]}), 计数 = {hist[i]}")

# plot
plt.figure(figsize=(10, 6))
plt.hist(radii, bins=list(bins), color='skyblue', edgecolor='black')
plt.title('Particle Radius Distribution', fontsize=24)
plt.xlabel('Radius', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 40)
plt.grid(True)

# mean std
plt.axvline(average_radius, color='r', linestyle='dashed', linewidth=1)
plt.axvline(average_radius - std_dev_radius, color='g', linestyle='dashed', linewidth=1)
plt.axvline(average_radius + std_dev_radius, color='g', linestyle='dashed', linewidth=1)
plt.legend(['Mean Radius', 'Std Deviation'])

plt.show()

print(average_radius)
print(std_dev_radius)

images = tiff.imread('5slices_RMS_1.tif')

# equivalent radius
radii2 = []
porosity = []

for i in range(images.shape[0]):
    # step2: label particles
    labeled_image, num_features = label(images[i], return_num=True)

    # step3: properties analysis
    properties = regionprops(labeled_image)
    porosity.append(np.sum(images[i] == 255) / (images[i].shape[0] * images[i].shape[1]))

    for prop in properties:
        # equivalent radius
        s = prop.area  # volume
        radius = (s / np.pi) ** (1/2)
        radii2.append(radius)

# mean and std
average_radius2 = float(np.mean(radii))
std_dev_radius2 = float(np.std(radii))
porosity_avg = np.mean(porosity)
# print(radii)
# bins = np.arange(0, 15.2, 0.2)
hist2, bin_edges2 = np.histogram(radii2, bins=bins)

similarity, _ = pearsonr(hist, hist2)
EMD = wasserstein_distance(hist, hist2)
print(EMD)
print(f"porosity: {porosity_avg}")
print(f"Similarity: {similarity}")
