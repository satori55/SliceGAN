import os
import tifffile
import einops
import numpy as np


def eachFile(path_file) -> list:
    """
    get file name from path_file
    :param path_file: target path
    :return: file name list in target path
    """
    fileName = []
    for file in os.listdir(path_file):
        if os.path.isfile(os.path.join(path_file, file)):
            fileName.append(file)
    return fileName


def slice_rank(slices) -> int:
    try:
        start_index = slices.find(".tif") + len(".tif")
    except:
        raise ValueError("The file name does not contain the keyword 'images'")

    try:
        end_index = slices.find(".tif", start_index)
    except:
        raise ValueError("The file name does not contain the keyword '.tif'")

    return int(slices[start_index:end_index])


# 定义2D TIFF切片的文件名列表
tiff_slices: list = eachFile("data")
sort_slices: list = sorted(tiff_slices, key=slice_rank)
print(sort_slices)

# 读取所有切片并存储在一个列表中
images: list = [tifffile.imread("data/" + slice_file) for slice_file in sort_slices]

image_crop:list = [element[107:363, 191:447, 1] for element in images]

# 将列表中的图像堆叠成一个3D数组
image_stack: np.ndarray = np.stack(image_crop, axis=0)

# image_stack = einops.rearrange(image_stack, "z y x -> y z x")

image_np: np.ndarray = image_stack[50]
image_np_transformed: np.ndarray = np.zeros_like(image_np)
image_np_transformed[(image_np >= 0) & (image_np < 128)] = 0
image_np_transformed[(image_np >= 128) & (image_np < 192)] = 128
image_np_transformed[(image_np >= 192) & (image_np <= 255)] = 255

# 将3D数组保存为一个新的3D TIFF文件
tifffile.imwrite("stacked_3d_image.tif", image_np_transformed)
