import cv2
import tifffile as tiff
from matplotlib import pyplot as plt


def connect_area(image_path: str) -> None:
    binary_img = tiff.imread(image_path)
    # calculate connected components
    num_labels, labels_im = cv2.connectedComponents(binary_img)
    labels_im_copy = labels_im.copy()
    # print(np.unique(labels_im))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_img, cmap="gray")
    plt.title("binary_img")
    plt.subplot(1, 2, 2)
    contra = labels_im_copy.copy()
    contra[contra == 0] = -20
    plt.imshow(contra, cmap="tab20b")
    plt.title("labels_im")
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2
    )
    plt.show()


if __name__ == "__main__":
    connect_area(
        f"D:/SliceGAN/ctdata/segment/1/20240430_YP202432019_Liu TT_powder_1#_1.0078um_manual_Export0001.tiff"
    )
