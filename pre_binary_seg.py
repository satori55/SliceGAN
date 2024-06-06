# %%
import os
import ants
import cv2
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
import tifffile as tiff


def normalize(img):
    """
    normalize image to 0-1
    :param img: input image
    :return: normalized image
    """
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img_normalized


def eachTif(path_folder):
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
            if os.path.splitext(child)[1] == '.tiff':
                # print(child)
                tif_img = tiff.imread(child)
                tif_img_normalized = normalize(tif_img)
                Tif_img_normalized.append(tif_img_normalized)
    return Filename_img, Tif_img_normalized


def eachFolder(path_folder):
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

# %%
# crop
folder_path = "D:/SliceGAN/ctdata/1"
file_list, img_list = eachTif(folder_path)
print(file_list)

for idx, img in enumerate(img_list):
    h, w = img.shape
    width = 340
    img = img[h//2-width:h//2+width, w//2-width:w//2+width]
    img = img * 255
    tiff.imwrite(f"D:/SliceGAN/ctdata/crop/1/{idx}.tif", img.astype(np.uint8))

# %%

# 读取CT图像
image_path = r"D:\SliceGAN\ctdata\1\20240430_YP202432019_Liu TT_powder_1#_1.0078um_manual_Export0001.tiff"
ct_image = ants.image_read(image_path)
ct_image = ants.resample_image(ct_image, (256, 256), 1, 0)
# temp = ct_image.numpy()
# ct_image = ants.from_numpy(temp/255)

# 进行图像平滑处理
ct_image_smoothed = ants.smooth_image(ct_image, sigma=1.0)

# 使用Atropos进行图像分割
segmentation = ants.atropos(a=ct_image_smoothed, x=ct_image_smoothed, m='[0.3,1x1]', c='[5,0]', i='KMeans[2]')

# 获取分割结果
segmented_image = segmentation['segmentation']

# 保存分割结果
# output_path = 'path_to_save_segmented_image/segmented_image.nii.gz'
# ants.image_write(segmented_image, output_path)

# 显示分割结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(ct_image.numpy(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image.numpy(), cmap='gray')

plt.show()


segmented_image = segmentation['segmentation'].numpy()

# 转换为二值图像
binary_image = np.where(segmented_image > 2, 255, 0).astype(np.uint8)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# 应用开运算
# opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# 标记连通区域
label_image = measure.label(binary_image)

cleaned_image = morphology.remove_small_objects(label_image, min_size=80)

# 显示分割连通区域的结果
plt.subplot(1, 2, 2)
plt.title("Connected Components")
plt.imshow(cleaned_image, cmap='nipy_spectral')

plt.show()

# %%
