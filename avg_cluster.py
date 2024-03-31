import numpy as np
import tifffile as tiff
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的二值图像作为示例
# 假设1代表我们感兴趣的像素（节点），0代表背景
image = np.array([
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1]
])

# image = tiff.imread('stacked_binary_images.tif')
# image = image[..., 1] / 255

# 显示图像
plt.imshow(image, cmap='gray')
plt.title('Binary Image')
plt.show()

# 将图像转换为图
G = nx.Graph()

# 遍历图像的每个像素
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] == 1:
            # 为图中的每个黑色像素添加一个节点
            G.add_node((i, j))
            # 检查四个方向的相邻像素，如果也是黑色，则添加一条边
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= i + di < image.shape[0] and 0 <= j + dj < image.shape[1]:
                    if image[i + di, j + dj] == 1:
                        G.add_edge((i, j), (i + di, j + dj))

# 计算平均聚类系数
# 计算平均聚类系数之前先检查图中是否有节点和边
if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    print("图中没有节点或边，无法计算平均聚类系数。")
else:
    avg_clustering_coefficient = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering_coefficient}")

