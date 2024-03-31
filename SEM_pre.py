import os

import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image


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


file_list = eachFile("C:/Users/SATORI/My drive/GNN/data/SEM Image/SEM Image")
# print(file_list)

images: list[np.ndarray] = [
    tiff.imread("C:/Users/SATORI/My drive/GNN/data/SEM Image/SEM Image/" + file)
    for file in file_list
]
print(len(images))

images_crop = [element[50:-50, 50:-50] for element in images]
print(images_crop[0].shape)

# plt.imshow(images_crop[23], cmap='gray')
# plt.show()

# save_process = [tiff.imwrite("D:/SliceGAN/SEM/" + file, images_crop[i]) for i, file in enumerate(file_list)]

# _,binary_image = cv2.threshold(images_crop[0], 80, 255, cv2.THRESH_BINARY)

binary_images = [
    cv2.threshold(images_crop[i], 80, 255, cv2.THRESH_BINARY)[1]
    for i in range(len(images_crop))
]
print(len(binary_images))
print(binary_images[0].shape)
# plt.imshow(binary_images[0], cmap='gray')
# plt.show()

# binary_images_save = [
#     tiff.imwrite("D:/SliceGAN/SEMbin/" + file, binary_images[i])
#     for i, file in enumerate(file_list)
# ]

binary_fix_images = [
    tiff.imread("D:/SliceGAN/SEMbin/" + file) for file in file_list[:18]
]

kernel = np.ones((3, 3), np.uint8)
binary_resize_images = [
    cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) for image in binary_fix_images
]
binary_resize_images = [
    cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) for image in binary_resize_images
]

binary_resize_images = [cv2.resize(image, (302, 210)) for image in binary_fix_images]

binary_resize_images = [
    cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)[1] for image in binary_resize_images
]

image_stack = np.stack(binary_resize_images, axis=0)
tiff.imwrite("stacked_SEM_18Images.tif", image_stack)
