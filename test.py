import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_image(
    image_path, save_path, width=320, vertical_offset=-20, horizontal_offset=0
):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 获取图像的尺寸
    h, w = image.shape
    print(h, w)

    image = image[
        h // 2 + horizontal_offset - width : h // 2 + horizontal_offset + width,
        w // 2 + vertical_offset - width : w // 2 + vertical_offset + width,
    ]
    cv2.imwrite(save_path, image)


def kmeans_segmentation_gray(image_path, k):
    # 读取灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 获取图像的尺寸
    h, w = image.shape
    print(h, w)

    width = 320
    vertical_offset = -20
    horizontal_offset = 0
    image = image[
        h // 2 + horizontal_offset - width : h // 2 + horizontal_offset + width,
        w // 2 + vertical_offset - width : w // 2 + vertical_offset + width,
    ]
    h, w = image.shape

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 将图像转换为二维数组
    image_2d = image.reshape(h * w, 1)
    # 将像素值转换为浮点数
    image_2d = np.float32(image_2d)

    # 定义KMeans的停止条件和次数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用KMeans算法
    _, labels, centers = cv2.kmeans(
        image_2d, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 将中心值转换为整数
    centers = np.uint8(centers)
    # 将每个像素的标签转换为中心值
    segmented_image = centers[labels.flatten()]
    # 将图像重新转换为原始的形状
    segmented_image = segmented_image.reshape(h, w)
    cata = np.unique(segmented_image)
    print(cata)

    extracted_label = 1
    binary_img = (segmented_image == (cata[extracted_label])).astype(np.uint8)

    # 计算连通区域
    num_labels, labels_im = cv2.connectedComponents(binary_img)
    labels_im_copy = labels_im.copy()
    # 除去较小的连通区域
    for i in range(1, num_labels):
        if np.sum(labels_im == i) < 400:
            binary_img[labels_im == i] = 0
            labels_im_copy[labels_im == i] = 0

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 对图像进行开运算
    binary_img = cv2.morphologyEx(np.float32(binary_img), cv2.MORPH_OPEN, kernel_1)
    binary_img = cv2.morphologyEx(np.float32(binary_img), cv2.MORPH_CLOSE, kernel_2)

    num_labels, labels_im = cv2.connectedComponents(binary_img.astype(np.uint8))
    
    # 除去较小的连通区域
    for i in range(1, num_labels):
        if np.sum(labels_im == i) < 5000:
            binary_img[labels_im == i] = 0
            labels_im_copy[labels_im == i] = 0
    
    # 显示原始图像和分割后的图像
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
    plt.imshow(labels_im_copy, cmap="Paired")
    plt.show()


# 示例使用
kmeans_segmentation_gray(
    "20240428_YP202432019_Liu TT_powder_2#_0.701um_manual_Export0001.tiff", 3
)
