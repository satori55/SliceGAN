import cv2
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
from sympy import per
import tifffile as tiff
from skimage.measure import label, regionprops
from scipy.stats import wasserstein_distance
from scipy.ndimage import zoom
import matplotlib.patches as mpatches


def normalize_list(data):
    total = sum(data)
    if total == 0:
        return [0] * len(data)  # 如果总和为0，避免除零错误
    return [x / total for x in data]


gen = False

if gen:
    scale_2 = True
else:
    scale_2 = False


ratio_0_7 = False
# ratio_0_7 = True

if gen:
    image_1 = tiff.imread(r"D:\SliceGAN\1_stack_20_scale2_192_2.tif")
    image_2 = tiff.imread(r"D:\SliceGAN\2_stack_20_scale2_192_2.tif")
    image_3 = tiff.imread(r"D:\SliceGAN\3_stack_20_scale2_192_2.tif")
    # image_4 = tiff.imread(r"D:\SliceGAN\4_stack_15_scale2_192_2.tif")
else:
    image_1 = tiff.imread(r"./gt_1.tif")
    image_2 = tiff.imread(r"./gt_2.tif")
    image_3 = tiff.imread(r"./gt_3.tif")
    # image_4 = tiff.imread(r"D:\SliceGAN\ctdata\fixed\4_stack_15.tif")

if gen:
    boxplot = []
else:
    boxplot_2 =[]

for idx, images in enumerate([image_1, image_2, image_3]):
    # images = tiff.imread(r"D:\SliceGAN\1_stack_20_scale2_192_2.tif")
    # images = tiff.imread(r"D:\SliceGAN\ctdata\fixed\1_stack_20.tif")
    phase = np.unique(images)
    # 使用scipy.ndimage.zoom进行插值，将尺寸放大n倍
    # zoom_factors = (5, 5, 5)
    # images = zoom(images, zoom_factors, order=0)
    # print("Interpolated shape:", images.shape)

    # eliminate small particles
    if gen:
        eliminate_area = [40, 35, 10] # 40, 35, 10, 20
    else:
        eliminate_area = [10, 10, 10]

    # porosity
    porosity_127 = []
    porosity_255 = []

    for i in range(images.shape[0]):
        area_127 = np.sum(images[i, ...] == phase[1])
        area_255 = np.sum(images[i, ...] == phase[2])
        porosity_127.append(1 - (area_127 / (images[i, ...].shape[0] * images[i, ...].shape[1])))
        porosity_255.append(1 - (area_255 / (images[i, ...].shape[0] * images[i, ...].shape[1])))

    average_porosity_127 = np.mean(porosity_127)
    average_porosity_255 = np.mean(porosity_255)
    print(f"{average_porosity_127=}")
    print(f"{average_porosity_255=}")

    # equivalent radius
    radii = []
    roundness = []
    aspect_ratio = []

    for i in range(images.shape[0]):
        # step2: label particles
        labeled_image, num_features = label(images[i], return_num=True)

        # step3: properties analysis
        properties = regionprops(labeled_image)

        for prop in properties:
            if prop.area <= eliminate_area[idx]:
                continue
            
            # equivalent radius roundness
            s = prop.area  # volume
            perimeter = prop.perimeter

            if perimeter == 0:
                roundness.append(0)
            else:
                roundness.append(4 * np.pi * s / perimeter ** 2)
                
            radius = (s / np.pi) ** (1/2)
            radii.append(radius)
            
            # 获取连通区域的轮廓
            coords = prop.coords
            contour = coords[:, [1, 0]].astype(np.int32)  # 转换为OpenCV格式
            
            # 获取最小旋转外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            # 计算最小旋转外接矩形的宽和高
            width_rot = np.linalg.norm(box[0] - box[1])
            height_rot = np.linalg.norm(box[1] - box[2])
            
            # 计算长宽比
            aspect_ratio_rot = width_rot / height_rot if width_rot < height_rot else height_rot / width_rot
            
            aspect_ratio.append(aspect_ratio_rot)
            # print(f"Aspect Ratio (rotated): {aspect_ratio_rot}")
        # print(f"finish{i}")
            
    # equivalent to 2 times the original data
    if ratio_0_7:
        radii = [r * 0.7 for r in radii]
    if scale_2:
        radii = [r * 2 for r in radii]

    print(f"{len(radii)=}")
    print(f"{len(roundness)=}")
    print(f"{radii=}")
    print(f"{roundness=}")
    percentiles = np.percentile(radii, [10, 50, 90])
    print(f"{percentiles=}")

    # mean and std
    average_radius = float(np.mean(radii))
    std_dev_radius = float(np.std(radii))
    # print(radii)
    bins = np.arange(0, 150, 2)
    # bins = [0, 10, 50, 90]
    hist, bin_edges = np.histogram(radii, bins=bins)

    # calculate the average roundness of each particle size in the bins
    roundness_bins = [[] for _ in range(len(hist))]
    for r, round in zip(radii, roundness):
        for i in range(len(hist)):
            if bin_edges[i] <= r < bin_edges[i + 1]:
                roundness_bins[i].append(round)
                break

    # remove the elements larger than 1 in the r of roundness_bins
    for i in range(len(roundness_bins)):
        roundness_bins[i] = [r for r in roundness_bins[i] if r <= 1]
        # print(max(roundness_bins[i]))
        
    # calculate the percentiles roundness
    aspect_ratio_filter = []
    for r, a in zip(roundness, aspect_ratio):
        if r <= 1:
            aspect_ratio_filter.append(a)

    hist_aspect_ratio = np.percentile(aspect_ratio_filter, [10, 50, 90])
    print(f"{hist_aspect_ratio=}")

    roundness = [r for r in roundness if r <= 1]
    percentiles_roundness = np.percentile(roundness, [10, 50, 90])
    print(f"{percentiles_roundness=}")

    # 打印每个分组的边界和计数
    for i in range(len(hist)):
        print(f"分组 {i+1}: 边界 = ({bin_edges[i]}, {bin_edges[i+1]}), 计数 = {hist[i]}")

    # plot
    plt.figure(figsize=(10, 6))
    # plt.hist(radii, bins=list(bins), color='skyblue', edgecolor='black')
    plt.plot(bin_edges[:-1], hist, linestyle='-', marker='o', color='skyblue')
    plt.title('Particle Radius Distribution', fontsize=24)
    plt.xlabel('Radius', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xlim(0, 15)
    plt.grid(True)

    # mean std
    plt.axvline(average_radius, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(average_radius - std_dev_radius, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(average_radius + std_dev_radius, color='g', linestyle='dashed', linewidth=1)
    plt.legend(['Mean Radius', 'Std Deviation'])
    plt.grid(axis='x')
    plt.show()

    print(f"{average_radius=}")
    print(f"{std_dev_radius=}")

    if gen:
        boxplot.append(aspect_ratio_filter)
    else:
        boxplot_2.append(aspect_ratio_filter)
    
    # 绘制箱线图
    plt.boxplot(radii, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='bisque', color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='royalblue'),
                flierprops=dict(marker='o', markeredgecolor='deepskyblue', markersize=4),
                medianprops=dict(color='lightcoral'),
                showfliers=False)

    plt.title('Particle Radius Distribution of Samples', fontsize=24)
    plt.xlabel('Radius', fontsize=20)
    plt.ylabel('Sample', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.show()

    images_2 = tiff.imread(r"D:\SliceGAN\ctdata\fixed\1_stack_20.tif")

    # equivalent radius
    radii2 = []

    for i in range(images_2.shape[0]):
        # step2: label particles
        labeled_image, num_features = label(images_2[i], return_num=True)

        # step3: properties analysis
        properties = regionprops(labeled_image)

        for prop in properties:
            if prop.area <= eliminate_area[idx]:
                continue
            # equivalent radius
            s = prop.area  # volume
            radius = (s / np.pi) ** (1/2)
            radii2.append(radius)
            
    if ratio_0_7:
        radii2 = [r * 0.7 for r in radii2]
    # mean and std
    average_radius2 = float(np.mean(radii2))
    std_dev_radius2 = float(np.std(radii2))
    # print(radii)
    # bins = np.arange(0, 150, 5)
    hist2, bin_edges2 = np.histogram(radii2, bins=bins)

    hist = normalize_list(hist)
    hist2 = normalize_list(hist2)
    EMD = wasserstein_distance(hist, hist2)
    print(f"{EMD=}")
    print(f"{hist=}")
    ratio_0_7 = True
    
    

new = []
for i, j in zip(boxplot, boxplot_2):
    new.append(i)
    new.append(j)

positions = [1, 2, 3.5, 4.5, 6, 7]
fig, ax = plt.subplots(figsize=(10, 7))
bp = ax.boxplot(new, positions = positions,
            vert=False,
            patch_artist=True, 
            boxprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='royalblue'),
            flierprops=dict(marker='o', markeredgecolor='deepskyblue', markersize=4),
            medianprops=dict(color='lightcoral'),
            showfliers=False)

# 设置每个箱体的颜色
colors = ['lightblue', 'bisque', 'lightblue', 'bisque', 'lightblue', 'bisque']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    
ax.set_yticks([1.5, 4, 6.5])
ax.set_yticklabels(['1', '2', '3'])

ax.set_title('Aspect Ratio Distribution of Samples', fontsize=22, fontname='Arial')
ax.set_xlabel('Aspect Ratio', fontsize=22, fontname='Arial')
ax.set_ylabel('Sample', fontsize=22, fontname='Arial')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.grid(axis='x')

legend_patches = [mpatches.Patch(color='bisque', label='Real Data'),
                  mpatches.Patch(color='lightblue', label='Generated Data')]

ax.legend(handles=legend_patches, loc='upper right', fontsize=14, bbox_to_anchor=(1.31, 1))

plt.savefig('Aspect_ratio.png', dpi=600, bbox_inches='tight')
plt.show()
