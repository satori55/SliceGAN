import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


def normalize(img) -> np.ndarray:
    """
    normalize image to 0-1
    :param img: input image
    :return: normalized image
    """
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img_normalized

def eachTif(path_folder) -> tuple:
    """
    get tif images from path_folder
    :param path_folder: the path of the tif image folder
    :return: image name list and image list
    """
    Filename_img = []
    Tif_img_normalized = []

    for file in os.listdir(path_folder):
        child = os.path.join(path_folder, file)
        if os.path.isdir(child):
            eachTif(child)
        else:
            Filename_img.append(file)
            if os.path.splitext(child)[1] == ".tif" or ".tiff":
                # print(child)
                tif_img = tiff.imread(child)
                tif_img_normalized = normalize(tif_img)
                Tif_img_normalized.append(tif_img_normalized)
    return Filename_img, Tif_img_normalized


def eachFolder(path_folder) -> list:
    """
    get folder name from path_folder
    :param path_folder: target path
    :return: folder name list in target path
    """
    folderName = []
    for folder in os.listdir(path_folder):
        if os.path.isdir(os.path.join(path_folder, folder)):
            folderName.append(folder)
    return folderName

# merge the label 1 and 2 of the #2
name_1, img_1 = eachTif("D:/SliceGAN/ctdata/fixed/2/label1")
name_2, img_2 = eachTif("D:/SliceGAN/ctdata/fixed/2/label2")

img_1 = np.array(img_1)
img_2 = np.array(img_2)
print(img_1.shape, img_2.shape)

# change the frontground of img_1 to 127
img_1[img_1 > 0] = 127
img_1 = img_1.astype(np.uint8)

# change the frontground of img_2 to 255
img_2[img_2 > 0] = 255
img_2 = img_2.astype(np.uint8)

# merge the img_1 and img_2
img_merge = img_1 + img_2

img_merge[img_merge == 126] = 127
print(np.unique(img_merge))

# save each slice
for i in range(img_merge.shape[0]):
    tiff.imsave("D:/SliceGAN/ctdata/fixed/2/label_merge/" + name_1[i], img_merge[i])
    