from tkinter import font
import cv2
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
from sympy import factor, per
import tifffile as tiff
from skimage.measure import label, regionprops
from scipy.stats import wasserstein_distance
from scipy.ndimage import zoom
from torch import le
import pandas as pd
from plotnine import ggplot, aes, geom_segment, geom_point, theme_minimal, labs, theme, element_rect, element_text, element_line, geom_line, scale_shape_manual, scale_x_continuous, element_blank
from scipy.ndimage import distance_transform_edt
from scipy.signal import wiener


def normalize_list(data):
    total = sum(data)
    if total == 0:
        return [0] * len(data)  # 如果总和为0，避免除零错误
    return [x / total for x in data]


# scale_2 = False
scale_2 = True

ratio_0_5 = False
ratio_0_7 = False
ratio_0_7 = True
# ratio_0_5 = True

image1_gt = tiff.imread(r"./ctdata/fixed/gt_7s.tif")
image1_gen = tiff.imread(r"D:\SliceGAN\7s_stack_50_scale2_192.tif")

# image1_gt = tiff.imread(r"./ctdata/fixed/gt_6.tif")
# image1_gen = tiff.imread(r"D:\SliceGAN\6_stack_30_scale2_192.tif")

# image1_gt = tiff.imread(r"./ctdata/fixed/gt_7f.tif")
# image1_gen = tiff.imread(r"./7f_stack_30_scale2_192.tif")

phase = np.unique(image1_gt)
print(phase)
# 使用scipy.ndimage.zoom进行插值，将尺寸放大n倍
# zoom_factors = (5, 5, 5)
# images = zoom(images, zoom_factors, order=0)
# print("Interpolated shape:", images.shape)

# eliminate small particles
eliminate_area = 10  # 1:40, 2:35, 3:10, 5:20, 6:20, 7s:10, 7f:20

"""
# distance transform
gen_distance_3d = np.zeros_like(images, dtype=float)

# 对每个2D切片进行距离变换
for z in range(images.shape[0]):
    foreground_2d = images[z, :, :].copy()
    foreground_2d[foreground_2d > 1] = 255
    distance_2d = distance_transform_edt(foreground_2d)
    gen_distance_3d[z, :, :] = distance_2d

print(f"Distance Transform Shape: {gen_distance_3d.shape}")
print(f"Max Distance: {np.max(gen_distance_3d)}")
print(f"Min Distance: {np.min(gen_distance_3d)}")
print(f"Mean Distance: {np.mean(gen_distance_3d)}")

# porosity
porosity_127 = []
# porosity_255 = []

for i in range(images.shape[0]):
    # print(np.unique(images[i, ...]))
    area_127 = np.sum(images[i, ...] == phase[1])
    # area_255 = np.sum(images[i, ...] == phase[2])
    porosity_127.append(1 - (area_127 / (images[i, ...].shape[0] * images[i, ...].shape[1])))
    # porosity_255.append(1 - (area_255 / (images[i, ...].shape[0] * images[i, ...].shape[1])))

average_porosity_127 = np.mean(porosity_127)
# average_porosity_255 = np.mean(porosity_255)
print(f"{average_porosity_127=}")
# print(f"{average_porosity_255=}")
"""

# equivalent radius
radii = []
roundness = []
aspect_ratio = []

for i in range(image1_gen.shape[0]):
    # step2: label particles
    labeled_image, num_features = label(image1_gen[i], return_num=True)

    # step3: properties analysis
    properties = regionprops(labeled_image)

    for prop in properties:
        if prop.area <= eliminate_area:
            continue

        # equivalent radius roundness
        s = prop.area  # volume
        perimeter = prop.perimeter

        if perimeter == 0:
            roundness.append(0)
        else:
            roundness.append(4 * np.pi * s / perimeter ** 2)

        radius = (s / np.pi) ** (1 / 2)
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
elif ratio_0_5:
    radii = [r * 0.5 for r in radii]

if scale_2:
    radii = [r * 2 for r in radii]

print(f"{len(radii)=}")
print(f"{len(roundness)=}")
# print(f"{radii=}")
# print(f"{roundness=}")
percentiles_radii = np.percentile(radii, [10, 50, 90])
print(f"{percentiles_radii=}")


# mean and std
average_radius = float(np.mean(radii))
std_dev_radius = float(np.std(radii))
# print(radii)
bins = np.arange(0, 84, 1)
# bins = [0, 10, 50, 90]
hist, bin_edges = np.histogram(radii * 2, bins=bins)

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
bins_roundness = np.arange(0, 1.1, 0.05)
# hist, bin_edges = np.histogram(roundness, bins=bins_roundness)


# 打印每个分组的边界和计数
for i in range(len(hist)):
    print(f"分组 {i + 1}: 边界 = ({bin_edges[i]}, {bin_edges[i + 1]}), 计数 = {hist[i]}")

"""
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
"""

"""
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
"""

# distance transform
gt_distance_3d = np.zeros_like(image1_gt, dtype=float)
# gt_distance_3d = gt_distance_3d[:, ::2, ::2]

# 对每个2D切片进行距离变换
for z in range(image1_gt.shape[0]):
    foreground_2d = image1_gt[z, :, :]
    foreground_2d[foreground_2d > 1] = 255
    distance_2d = distance_transform_edt(foreground_2d)
    gt_distance_3d[z, :, :] = distance_2d

print(f"Distance Transform Shape: {gt_distance_3d.shape}")
print(f"Max Distance: {np.max(gt_distance_3d)}")
print(f"Min Distance: {np.min(gt_distance_3d)}")
print(f"Mean Distance: {np.mean(gt_distance_3d)}")

# equivalent radius
radii2 = []
roundness2 = []
aspect_ratio2 = []

for i in range(image1_gt.shape[0]):
    # step2: label particles
    labeled_image, num_features = label(image1_gt[i], return_num=True)

    # step3: properties analysis
    properties = regionprops(labeled_image)

    for prop in properties:
        if prop.area <= 10:
            continue
        # equivalent radius
        s = prop.area  # volume
        radius = (s / np.pi) ** (1 / 2)
        radii2.append(radius)

        # equivalent radius roundness
        perimeter = prop.perimeter

        if perimeter == 0:
            roundness2.append(0)
        else:
            roundness2.append(4 * np.pi * s / perimeter ** 2)

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

        aspect_ratio2.append(aspect_ratio_rot)

if ratio_0_7:
    radii2 = [r * 0.7 for r in radii2]
elif ratio_0_5:
    radii2 = [r * 0.5 for r in radii2]

# mean and std
average_radius2 = float(np.mean(radii2))
std_dev_radius2 = float(np.std(radii2))
# print(radii)
# bins = np.arange(0, 150, 5)
hist2, bin_edges2 = np.histogram(radii2 * 2, bins=bins)

roundness2 = [r for r in roundness2 if r <= 1]

aspect_ratio_filter2 = []
for r, a in zip(roundness2, aspect_ratio2):
    if r <= 1:
        aspect_ratio_filter2.append(a)

# hist2, bin_edges2 = np.histogram(roundness2, bins=bins_roundness)

hist = normalize_list(hist)
hist2 = normalize_list(hist2)


EMD_psd = wasserstein_distance(hist, hist2)
print(f"{EMD_psd=}")
print(f"{hist=}")


window_size = 5
window = np.ones(window_size) / window_size
hist_smooth_Gen = np.convolve(hist, window, mode='same')
hist_smooth_real = np.convolve(hist2, window, mode='same')

for idx, item in enumerate(hist):
    if item == 0:
        hist_smooth_Gen[idx] = 0

for idx, item in enumerate(hist2):
    if item == 0:
        hist_smooth_real[idx] = 0

# keep the beginning and the end to be zero
hist_smooth_Gen[0] = 0
hist_smooth_Gen[-1] = 0
hist_smooth_real[0] = 0
hist_smooth_real[-1] = 0

# hist_smooth = hist
# hist2_smooth = hist2

hist_smooth_Gen_hat = hist_smooth_Gen
hist_smooth_real_hat = hist_smooth_real

percentiles_radii2 = np.percentile(radii2, [10, 50, 90])
print(f"{percentiles_radii2=}")
percentiles_roundness2 = np.percentile(roundness2, [10, 50, 90])
print(f"{percentiles_roundness2=}")
percentiles_aspect_ratio2 = np.percentile(aspect_ratio2, [10, 50, 90])
print(f"{percentiles_aspect_ratio2=}")


# """
# plot
plt.figure(figsize=(8, 5))

plt.gca().spines['top'].set_linewidth(2)     # 上边框线宽
plt.gca().spines['bottom'].set_linewidth(2)  # 下边框线宽
plt.gca().spines['left'].set_linewidth(2)    # 左边框线宽
plt.gca().spines['right'].set_linewidth(2)   # 右边框线宽

# plt.hist(radii, bins=list(bins), color='skyblue', edgecolor='black')
# color: #7CCD7C, #43CD80, wheat, peru, skyblue, lightblue
# plt.plot(bin_edges[:-1], hist_smooth_Gen_hat * 100, linestyle='--', marker=None, color='lightblue', linewidth=3, label='Generated fast scan')
# plt.plot(bin_edges[:-1], hist_smooth_real_hat * 100, linestyle='-', marker=None, color='skyblue', linewidth=3, label='Real fast scan')
# plt.plot(bin_edges[:-1], hist_smooth_Gen * 100, linestyle='--', marker=None, color='wheat', linewidth=3, label='Generated slow scan')
# plt.plot(bin_edges[:-1], hist_smooth_real * 100, linestyle='-', marker=None, color='peru', linewidth=3, label='Real slow scan')

###
plt.plot(bin_edges[:-1], hist_smooth_Gen_hat * 100, linestyle='--', marker=None, color='#7CCD7C', linewidth=3, label='Generated #1')
plt.plot(bin_edges[:-1], hist_smooth_real_hat * 100, linestyle='-', marker=None, color='#43CD80', linewidth=3, label='Real #1')
plt.plot(bin_edges[:-1], hist_smooth_Gen * 100, linestyle='--', marker=None, color='wheat', linewidth=3, label='Generated #2')
plt.plot(bin_edges[:-1], hist_smooth_real * 100, linestyle='-', marker=None, color='peru', linewidth=3, label='Real #2')


# plt.title('Particle Aspect Ratio Distribution', fontsize=22, fontname="Arial")
# plt.fill_between(bin_edges[:-1], hist_smooth_hat * 100, color='#7CCD7C', alpha=0.6, label='Generated #7 fast')
# plt.fill_between(bin_edges[:-1], hist_smooth * 100, color='wheat', alpha=0.6, label='Generated #7 slow')

# plt.xlabel('Roundness', fontsize=26, fontname="Arial", fontweight='bold')
plt.xlabel('Diameter(μm)', fontsize=26, fontname="Arial", fontweight='bold')
# plt.xlabel('Aspect Ratio', fontsize=26, fontname="Arial", fontweight='bold')

plt.ylabel('Frequency (%)', fontsize=26, fontname="Arial", fontweight='bold')
plt.xticks(fontsize=26, fontname="Arial", fontweight='bold')
plt.yticks(fontsize=26, fontname="Arial", fontweight='bold')
plt.tick_params(axis='both', width=3, length=6)  # width控制粗细，length控制长度

plt.legend(loc='upper right', prop={'size': 20, 'family': 'Arial', 'weight': 'bold'}, frameon=False)
plt.ylim(0, 15)
# plt.xscale('log')
# plt.xticks([1, 10, 100])  # 自定义刻度：0.1, 10, 100
# plt.xlim(1, 100)
# plt.grid(True)
# plt.grid(axis='x')
plt.savefig("67Particle_psd.png", dpi=600, bbox_inches='tight')
plt.show()

# """

"""
print(len(hist), len(hist2))

# Similarity of the two distributions
coff_psd = np.corrcoef(hist, hist2)
print(f"{coff_psd=}")

bins_aspect_ratio = np.arange(0, 1.1, 0.1)
hist_aspect_ratio = np.histogram(aspect_ratio_filter, bins=bins_aspect_ratio)
hist_aspect_ratio = normalize_list(hist_aspect_ratio[0])
hist_aspect_ratio2 = np.histogram(aspect_ratio_filter2, bins=bins_aspect_ratio)
hist_aspect_ratio2 = normalize_list(hist_aspect_ratio2[0])

bins_roundness = np.arange(0, 1.1, 0.1)
hist_roundness = np.histogram(roundness, bins=bins_roundness)
hist_roundness = normalize_list(hist_roundness[0])
hist_roundness2 = np.histogram(roundness2, bins=bins_roundness)
hist_roundness2 = normalize_list(hist_roundness2[0])

print(len(hist_aspect_ratio), len(hist_roundness))

coff_aspect_ratio = np.corrcoef(hist_aspect_ratio, hist_aspect_ratio2)
coff_roundness = np.corrcoef(hist_roundness, hist_roundness2)
print(f"{coff_aspect_ratio=}")
print(f"{coff_roundness=}")

EMD_roundess = wasserstein_distance(hist_roundness, hist_roundness2)
EMD_aspect_ratio = wasserstein_distance(hist_aspect_ratio, hist_aspect_ratio2)
print(f"{EMD_roundess=}")
print(f"{EMD_aspect_ratio=}")
"""

# geom hist plot


# 创建示例数据
# data = {
#     'category': bin_edges[:-1],
#     'start': hist,
#     'end': hist2
# }
# df = pd.DataFrame(data)

# 创建竖直方向的杠铃图
# plot = (ggplot(df, aes(y='start', yend='end', x='category', xend='category'))
#         + geom_segment(size=1, color='#D3D3D3')
#         + geom_point(aes(y='start'), color='#006D2C', size=2)
#         + geom_point(aes(y='end'), color='#B2DF8A', size=2)
#         + theme_minimal()
#         + theme(panel_background=element_rect(fill='white', color='white'),
#                 plot_background=element_rect(fill='white', color='white'))
#         + labs(title='ggplot2 geom_dumbbell with dot guide'))

# # 显示图表
# print(plot)


# 创建数据框架
# df1 = pd.DataFrame({
#     'count': hist,
#     'bin_mid': 0.5 * (bin_edges[1:] + bin_edges[:-1]),
#     'group': 'Generated'
# })

# df2 = pd.DataFrame({
#     'count': hist2,
#     'bin_mid': 0.5 * (bin_edges[1:] + bin_edges[:-1]),
#     'group': 'Real'
# })

# # 合并数据框架
# df = pd.concat([df1, df2])

# 创建棒棒糖图
# plot = (ggplot(df, aes(x='bin_mid', y='count', color='group'))
#         + geom_segment(aes(x='bin_mid', xend='bin_mid', y=0, yend='count'), size=1)
#         + geom_point(size=3)
#         + theme_minimal()
#         + theme(panel_background=element_rect(fill='white', color='white'),
#                 plot_background=element_rect(fill='white', color='white'))
#         + labs(title='PSD', x='Radius', y='Frequency'))

# print(plot)

# x_min = bin_edges.min()
# x_max = bin_edges.max()

# 创建折线图
# plot = (ggplot(df, aes(x='bin_mid', y='count', color='group', shape='group'))
#         + geom_line(aes(group='group'), size=2)
#         + geom_point(size=3)
#         + theme_minimal()
#         + theme(
#             panel_background=element_rect(fill='white', color='white'),
#             plot_background=element_rect(fill='white', color='white'),
#             panel_grid_major=element_blank(),  # 去除主要网格线
#             panel_grid_minor=element_blank(),  # 去除次要网格线
#             panel_border=element_rect(color='black', fill=None),  # 添加四周框线
#             axis_line=element_line(color='black'),  # 添加轴线
#             axis_ticks_major=element_line(color='black', size=0.75),  # 添加主要刻度线
#             # axis_ticks_minor=element_line(color='black', size=0.5),   # 添加次要刻度线
#             axis_ticks_length_major=6,  # 主要刻度线长度
#             # axis_ticks_length_minor=4,  # 次要刻度线长度
#             text=element_text(size=15),
#             axis_title=element_text(size=20),
#             plot_title=element_text(size=24),
#             legend_position=(0.95, 0.95),
#             legend_justification=(1, 1),
#             axis_text_x=element_text(size=22, margin={'t': 10}),  # 调整X轴文字与轴线的距离
#             axis_text_y=element_text(size=22, margin={'r': 10})   # 调整Y轴文字与轴线的距离
#         )
#         + labs(title='Particle Radius Distribution', x='Radius', y='Frequency')
#         + scale_shape_manual(values={'Generated': 'o', 'Real': 's'})  # 'o'是圆形，'s'是方形
#         + scale_x_continuous(breaks=np.arange(np.floor(x_min), np.ceil(x_max) + 1, 10))  # 横轴以10为距离，并显示两端值
#         + labs(color='', shape='')
# )


# plot.save("particle_radius_distribution.png", width=11, height=6, dpi=600)

print("Finish")
