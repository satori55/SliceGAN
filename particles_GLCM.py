from email.mime import image
from math import e
from cv2 import mean
import numpy as np
import tifffile as tiff
from skimage.feature.texture import graycomatrix, graycoprops


scale_2 = False
# scale_2 = True

images = tiff.imread(r"D:\SliceGAN\5_stack_20_scale1_192_2.tif")
images_2 = tiff.imread(r"./gt_5.tif")

phase_1 = np.unique(images)
map_dict = {phase_1[0]: 0, phase_1[1]: 1, phase_1[2]: 2}
images = np.vectorize(map_dict.get)(images)
print(np.unique(images))

phase_2 = np.unique(images_2)
map_dict_2 = {phase_2[0]: 0, phase_2[1]: 1, phase_2[2]: 2}
images_2 = np.vectorize(map_dict_2.get)(images_2)
print(np.unique(images_2))

for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
    print(f"Angle: {angle}")
    
    contrast_list = []
    homogeneity_list = []
    energy_list = []

    contrast_list_2 = []
    homogeneity_list_2 = []
    energy_list_2 = []

    for k in range(images.shape[0]):
        slice_2d = images[k, :, :]
        glcm = graycomatrix(
            slice_2d, distances=[1], angles=[angle], levels=3, symmetric=True, normed=True
        )

        contrast = graycoprops(glcm, "contrast")
        homogeneity = graycoprops(glcm, "homogeneity")
        energy = graycoprops(glcm, "energy")

        contrast_list.append(contrast)
        homogeneity_list.append(homogeneity)
        energy_list.append(energy)

    mean_contrast = np.mean(contrast_list)
    mean_homogeneity = np.mean(homogeneity_list)
    mean_energy = np.mean(energy_list)

    print(f"Average Contrast: {mean_contrast}")
    print(f"Average Homogeneity: {mean_homogeneity}")
    print(f"Average Energy: {mean_energy}")

    for k2 in range(images_2.shape[0]):
        if scale_2:
            slice_2d_2 = images_2[k2, ::2, ::2]
        else:
            slice_2d_2 = images_2[k2, :, :]
        glcm_2 = graycomatrix(
            slice_2d_2,
            distances=[1],
            angles=[angle],
            levels=3,
            symmetric=True,
            normed=True,
        )

        contrast_2 = graycoprops(glcm_2, "contrast")
        homogeneity_2 = graycoprops(glcm_2, "homogeneity")
        energy_2 = graycoprops(glcm_2, "energy")

        contrast_list_2.append(contrast_2)
        homogeneity_list_2.append(homogeneity_2)
        energy_list_2.append(energy_2)

    mean_contrast_2 = np.mean(contrast_list_2)
    mean_homogeneity_2 = np.mean(homogeneity_list_2)
    mean_energy_2 = np.mean(energy_list_2)

    print(f"Average Contrast 2: {mean_contrast_2}")
    print(f"Average Homogeneity 2: {mean_homogeneity_2}")
    print(f"Average Energy 2: {mean_energy_2}")
    print("\n")
    