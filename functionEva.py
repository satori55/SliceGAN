from email.mime import image
import re
import cv2
from matplotlib.mlab import angle_spectrum
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
from openpyxl import load_workbook


def normalize_list(data):
    total = sum(data)
    if total == 0:
        return [0] * len(data)  # 如果总和为0，避免除零错误
    return [x / total for x in data]


# Distribution of the radii, roundness, and aspect ratio of the particles
def RRA_dist(data_path: str, imaging_scale: float|bool=False, gen_scale: float|bool=False, eliminate: float|bool=False):
    # data_path = r"./ctdata/fixed/gt_7s.tif"
    if isinstance(data_path, str):
        image = tiff.imread(data_path)
        phase = np.unique(image)
        print(phase)
    else:
        image = data_path

    radii = []
    roundness = []
    aspect_ratio = []
    for i in range(image.shape[0]):
        # step2: label particles
        labeled_image, num_features = label(image[i], return_num=True)

        # step3: properties analysis
        properties = regionprops(labeled_image)

        for prop in properties:
            if eliminate is False:
                pass
            else:
                if prop.area <= eliminate:
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

    if gen_scale is False:
        pass
    else:
        radii = [r * gen_scale for r in radii]

    if imaging_scale is False:
        pass
    else:
        radii = [r * imaging_scale for r in radii]

    aspect_ratio_filter = []
    for r, a in zip(roundness, aspect_ratio):
        if r <= 1:
            aspect_ratio_filter.append(a)

    roundness = [r for r in roundness if r <= 1]

    # print(f"{len(radii)=}")
    # print(f"{len(roundness)=}")
    # print(f"{len(aspect_ratio_filter)=}")

    return radii, roundness, aspect_ratio_filter


if __name__ == "__main__":
    data_path = r"D:\SliceGAN\ctdata\fixed\6_stack_20_largest.tif"

    image = tiff.imread(data_path)
    radii, roundness, aspect = RRA_dist(image[0:10, ...], imaging_scale=0.7, gen_scale=1, eliminate=20)

    # radii_image = []
    # for i in range(image.shape[0]):
    #     data = image[i]
    #     # add an extra dimension
    #     data = np.expand_dims(data, axis=0)
    #     # print(data.shape)
    #     radii, roundness, aspect = RRA_dist(data, imaging_scale=0.7, gen_scale=2, eliminate=10)
    #     # print(np.mean(radii))
    #     # print(np.mean(roundness))
    #     # print(np.mean(aspect))
    #     radii_image.append(np.median(radii))

    # find out the index of the largest 10 elements in list radii_image
    # Convert the list to a numpy array for easy manipulation
    # print(len(radii_image))
    # radii_array = np.array(radii_image)
    # print(radii_array.shape)

    # # Find the indices of the largest 10 elements
    # largest_indices = np.argsort(radii_array)[-20:][::-1]  # Sorted in descending order
    # print(largest_indices)

    # empty_image = np.zeros_like(image)
    # for idx, i in enumerate(largest_indices):
    #     empty_image[idx] = image[i]

    # print(empty_image.shape)
    # tiff.imwrite(r"./6_stack_20_largest.tif", empty_image)




    # find out the 10,50,90 percentile of the radii
    radii = np.array(radii)
    roundness = np.array(roundness)
    aspect = np.array(aspect)

    print(f"{np.percentile(radii, 10)=}")
    print(f"{np.percentile(radii, 50)=}")
    print(f"{np.percentile(radii, 90)=}")

    print(f"{np.percentile(roundness, 10)=}")
    print(f"{np.percentile(roundness, 50)=}")
    print(f"{np.percentile(roundness, 90)=}")

    print(f"{np.percentile(aspect, 10)=}")
    print(f"{np.percentile(aspect, 50)=}")
    print(f"{np.percentile(aspect, 90)=}")

    bin_radii = np.arange(0, 150, 2)
    bin_roundness = np.arange(0, 1.1, 0.1)
    bin_aspect = np.arange(0, 1.1, 0.1)

    hist_radii, _ = np.histogram(radii, bins=bin_radii)
    hist_roundness, _ = np.histogram(roundness, bins=bin_roundness)
    hist_aspect, _ = np.histogram(aspect, bins=bin_aspect)

    print(f"{len(hist_roundness)=}")

    # normalize the histogram
    hist_radii = normalize_list(hist_aspect)

    wb = load_workbook(r"./Eva result.xlsx")
    ws = wb["Aspect"]
    for i in range(len(hist_radii)):
        ws.cell(row=i + 3, column=16).value = hist_radii[i]

    wb.save(r"./Eva result.xlsx")


    # plot Gen, real
    plt.figure(figsize=(10, 8))

    plt.gca().spines['top'].set_linewidth(2)     # 上边框线宽
    plt.gca().spines['bottom'].set_linewidth(2)  # 下边框线宽
    plt.gca().spines['left'].set_linewidth(2)    # 左边框线宽
    plt.gca().spines['right'].set_linewidth(2)   # 右边框线宽

    # plt.hist(radii, bins=list(bins), color='skyblue', edgecolor='black')
    # color: #7CCD7C, #43CD80, wheat, peru, skyblue, lightblue
    plt.plot(bin_edges[:-1], hist_smooth_Gen_hat * 100, linestyle='--', marker=None, color='lightblue', linewidth=2, label='Generated #7 fast')
    plt.plot(bin_edges[:-1], hist_smooth_real_hat * 100, linestyle='-', marker=None, color='skyblue', linewidth=2, label='Real #7 fast')
    plt.plot(bin_edges[:-1], hist_smooth_Gen * 100, linestyle='--', marker=None, color='wheat', linewidth=2, label='Generated #7 slow')
    plt.plot(bin_edges[:-1], hist_smooth_real * 100, linestyle='-', marker=None, color='peru', linewidth=2, label='Real #7 slow')
    # plt.title('Particle Aspect Ratio Distribution', fontsize=22, fontname="Arial")
    # plt.fill_between(bin_edges[:-1], hist_smooth_hat * 100, color='#7CCD7C', alpha=0.6, label='Generated #7 fast')
    # plt.fill_between(bin_edges[:-1], hist_smooth * 100, color='wheat', alpha=0.6, label='Generated #7 slow')

    plt.xlabel('Roundness', fontsize=22, fontname="Arial")
    # plt.xlabel('Diameter(μm)', fontsize=22, fontname="Arial")
    # plt.xlabel('Aspect Ratio', fontsize=22, fontname="Arial")

    plt.ylabel('Frequency (%)', fontsize=22, fontname="Arial")
    plt.xticks(fontsize=22, fontname="Arial")
    plt.yticks(fontsize=22, fontname="Arial")
    # plt.legend(prop={'size': 22, 'family': 'Arial'}, frameon=False)
    plt.ylim(0, 15)
    # plt.xscale('log')
    # plt.xticks([1, 10, 100])  # 自定义刻度：0.1, 10, 100
    # plt.xlim(1, 100)
    # plt.grid(True)
    # plt.grid(axis='x')
    plt.savefig("7Particle_roundness.png", dpi=600, bbox_inches='tight')
    plt.show()
