import numpy as np
import tifffile as tiff
from scipy.ndimage import distance_transform_edt as distance_transform
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def two_point_correlation(image, step_size=1, max_distance=None):
    """
    Calculate the two-point correlation function of a binary image.

    Parameters:
    - image: A binary (2D) numpy array where 1s represent the feature of interest.
    - step_size: Distance step size for sampling.
    - max_distance: Maximum distance to compute the correlation for. Defaults to the half of the image diagonal.

    Returns:
    - distances: Array of distances.
    - correlation: Two-point correlation function values at the distances.
    """

    # Ensure image is binary
    image = (image > 0).astype(np.int32)

    # Calculate the distance transform
    distance = distance_transform(1 - image)

    # Maximum distance for correlation
    if max_distance is None:
        max_distance = np.hypot(*image.shape) / 2

    # Sample distances
    distances = np.arange(0, max_distance, step_size)
    correlation = []

    # Calculate correlation for each distance
    for d in distances:
        mask = np.logical_and(distance > d - step_size / 2, distance <= d + step_size / 2)
        if image[mask].size == 0:
            correlation.append(0)
        else:
            correlation.append(image[mask].size / image.size)

    return distances, correlation

# Example usage:
if __name__ == "__main__":
    # Create a random binary image
    # np.random.seed(0)
    # image = np.random.choice([0, 1], size=(100, 100), p=[0.7, 0.3])
    image = tiff.imread('2_stack_20_scale2_192_2.tif')
    image = image[150, ...] / 255
    print(image.shape)
    distances, correlation = two_point_correlation(image, max_distance=10)

    # Plot the two-point correlation function
    plt.figure(figsize=(8, 5))
    plt.plot(distances, correlation)
    plt.xlabel('Distance')
    plt.ylabel('Correlation')
    plt.title('Two-Point Correlation Function')
    plt.show()
