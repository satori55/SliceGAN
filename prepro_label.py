import os
import cv2
from debugpy import connect
from matplotlib.pylab import f
from networkx import center
import numpy as np
import matplotlib.pyplot as plt
from sympy import plot_implicit
import tifffile as tiff


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
            if os.path.splitext(child)[1] == ".tiff":
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


def crop_image(
    image: np.ndarray,
    save_path: str | None = None,
    width=320,
    vertical_offset=-20,
    horizontal_offset=0,
) -> None:
    """crop the image

    Args:
        image (np.ndarray): image to be cropped
        save_path (str): the path to save the cropped image
        width (int, optional): the half width of the cropped image. Defaults to 320.
        vertical_offset (int, optional): the vertical offset of the cropped image. Defaults to -20.
        horizontal_offset (int, optional): the horizontal offset of the cropped image. Defaults to 0.
    """

    # 获取图像的尺寸
    h, w = image.shape
    # print(h, w)

    image = image[
        h // 2 + horizontal_offset - width : h // 2 + horizontal_offset + width,
        w // 2 + vertical_offset - width : w // 2 + vertical_offset + width,
    ]
    if save_path is not None:
        cv2.imwrite(save_path, image)


def kmeans_segmentation_gray(
    image_path: str | None = None,
    save_path: str | None = None,
    k: int = 2,
    connect_size_1=None,
    connect_size_2=None,
    kernel_1=None,
    kernel_2=None,
    plot_label=(0, True),
) -> None:
    """segmentation using kmeans

    Args:
        image_path (str): the path of the image
        save_path (str): the path to save the segmented image
        k (int): the number of clusters
        connect_size_1 ([type], optional): the size of the connected area to be removed. Defaults to None.
        connect_size_2 ([type], optional): the size of the connected area to be removed. Defaults to None.
        kernel_1 ([type], optional): the kernel for morphological operation. Defaults to None.
        kernel_2 ([type], optional): the kernel for morphological operation. Defaults to None.
        plot_label (tuple, optional): the label to plot and whether to show the figure. Defaults to (0, True).
    """

    image = tiff.imread(image_path)
    h, w = image.shape

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 将图像转换为二维数组
    image_2d = image.reshape(h * w, 1)
    # 将像素值转换为浮点数
    image_2d = image_2d.astype(np.float32)

    # 定义KMeans的停止条件和次数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用KMeans算法
    _, labels, centers = cv2.kmeans(
        image_2d * 50000, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # 将labels中的标签替换为centers中的中心值
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    # 将图像重新转换为原始的形状
    segmented_image = segmented_image.reshape(h, w)
    cluster = np.unique(segmented_image)
    print(cluster)

    extracted_label, plot_figure = plot_label
    binary_img = (segmented_image == (cluster[extracted_label])).astype(np.uint8)

    # 计算连通区域
    num_labels, labels_im = cv2.connectedComponents(binary_img)
    labels_im_copy = labels_im.copy()
    # print(np.unique(labels_im))

    # 除去较小的连通区域
    if connect_size_1 is not None:
        for i in range(1, num_labels):
            if np.sum(labels_im == i) < connect_size_1:
                binary_img[labels_im == i] = 0
                labels_im_copy[labels_im == i] = 0

    # 对图像进行开运算
    if kernel_1 is not None:
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_1)

    if kernel_2 is not None:
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_2)

    num_labels, labels_im = cv2.connectedComponents(binary_img.astype(np.uint8))
    labels_im_copy = labels_im.copy()

    # 除去较小的连通区域
    if connect_size_2 is not None:
        for i in range(1, num_labels):
            if np.sum(labels_im == i) < connect_size_2:
                binary_img[labels_im == i] = 0
                labels_im_copy[labels_im == i] = 0

    if save_path is not None:
    # print(np.unique(labels_im_copy))
        tiff.imwrite(save_path, binary_img * 255)
    
    # 显示原始图像和分割后的图像
    if plot_figure:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.subplot(2, 2, 2)
        plt.imshow(segmented_image, cmap="gray")
        plt.title("Segmented Image with K = {}".format(k))
        plt.subplot(2, 2, 3)
        plt.imshow(binary_img, cmap="gray")
        plt.title("Extracted Label = {}".format(extracted_label))
        plt.subplot(2, 2, 4)
        contr = labels_im_copy.copy()
        contr[contr == 0] = -20
        plt.imshow(contr, cmap="tab20b")

        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2
        )
        plt.show()


# 示例使用
class preprocess(object):
    def __init__(self, root: str, crop_save_path: str, seg_save_path: str):
        """

        Args:
            root (str): root path of the raw image
            crop_save_path (str): path to save the cropped image
            seg_save_path (str): path to save the segmented image
        """
        self.root = root
        self.img_name, self.img_list = eachTif(self.root)
        self.crop_save_path = crop_save_path
        self.seg_save_path = seg_save_path

    def crop(self, width=320, vertical_offset=-20, horizontal_offset=0):
        for name, img in zip(self.img_name, self.img_list):
            crop_image(
                image=img.astype(np.float32),
                save_path=f"{self.crop_save_path}/{name}",
                width=width,
                vertical_offset=vertical_offset,
                horizontal_offset=horizontal_offset,
            )

    def segment(
        self,
        k: int,
        connect_size_1=None,
        connect_size_2=None,
        kernel_1=None,
        kernel_2=None,
        plot_label=(0, True),
    ):
        for name, img in zip(self.img_name, self.img_list):
            kmeans_segmentation_gray(
                image_path=f"{self.crop_save_path}/{name}",
                save_path=f"{self.seg_save_path}/{name}", # comment for not saving the segmented image
                k=k,
                connect_size_1=connect_size_1,
                connect_size_2=connect_size_2,
                kernel_1=kernel_1,
                kernel_2=kernel_2,
                plot_label=plot_label,
            )
            # break


# 实例化类
if __name__ == "__main__":
    dataset_idx = 3
    
    connect_size_1 = 200
    connect_size_2 = 300
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # binary image open
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # binary image close
    
    # (extracted_label: choose the label to plot, plot_figure: show the figure or not)
    plot_label = (1,True)

    dataset = preprocess(
        root=f"D:/SliceGAN/ctdata/raw/{dataset_idx}",
        crop_save_path=f"D:/SliceGAN/ctdata/crop/{dataset_idx}",
        seg_save_path=f"D:/SliceGAN/ctdata/segment/{dataset_idx}",
    )
    # dataset.crop(width=320, vertical_offset=-20, horizontal_offset=0)
    dataset.segment(
        k=2, # for data1 and data3, the k cluster is 2, for data2, the k cluster is 3
        connect_size_1=connect_size_1,
        connect_size_2=connect_size_2,
        kernel_1=kernel_1,
        kernel_2=kernel_2,
        plot_label=plot_label,
    )