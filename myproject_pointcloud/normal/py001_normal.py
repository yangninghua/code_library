import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import numpy as np
np.set_printoptions(suppress=True)
import open3d as o3d

# my envs
# env:environment/mmlab_point3d_open3d/bin/python  open3d==0.9.0.0  numpy==1.19.5
# env:environment/miniconda3_py17/bin/python  open3d==0.13.0  numpy==1.21.0

# 以下代码是open3d==0.13.0.0

path = os.path.join(ROOT, "bun000.ply")
normalPath = path.replace("bun000.ply", "bun000_normal.ply")
pcd = o3d.io.read_point_cloud(path)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
print(pcd)
print(np.asarray(pcd.points))

#o3d.visualization.draw_geometries([pcd], "Open3D origin points", point_show_normal=False)

# 下采样滤波，体素边长为0.002m
downpcd = pcd.voxel_down_sample(voxel_size=0.004)
print(downpcd)
#o3d.visualization.draw_geometries([downpcd], "Open3D downsample points", point_show_normal=False)

if True:
    # 计算法线，只考虑邻域内的20个点
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20),
        fast_normal_computation=False)
else:
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.01),
        fast_normal_computation=False)

# 可视化法线
o3d.visualization.draw_geometries([downpcd], "Open3D normal estimation", point_show_normal=True)

# 输出前10个点
print(np.asarray(downpcd.points)[:10, :])
# 输出前10个点的法向量
print(np.asarray(downpcd.normals)[:10, :])

# 可视化法向量的点，并存储法向量点到文件
# 点云法向量的点都以绿色显示
normal_point = o3d.utility.Vector3dVector(downpcd.normals)
normals = o3d.geometry.PointCloud()
normals.points = normal_point
normals.paint_uniform_color((0, 1, 0))
o3d.visualization.draw_geometries([downpcd, normals], "Open3D noramls points")
o3d.io.write_point_cloud(normalPath, normals)


import open3d as o3d 
import numpy as np
from scipy.spatial import KDTree
def PCA(data, sort=True):
    # normalize 归一化
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    # 计算对称的协方差矩阵
    H = np.dot(normal_data.T, normal_data)
    # SVD奇异值分解，得到H矩阵的特征值和特征向量
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)
    if sort:
        #从大到小排序特征值
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

downpcd_np = np.array(downpcd.points)
tree = KDTree(downpcd_np)

if True:
    top_k = 20
    scipy_neighbor_dist, scipy_neighbor_idx = tree.query(downpcd_np, top_k)
    scipy_normals = []
else:
    radius = 0.1
    scipy_neighbor_idx = tree.query_ball_point(downpcd_np, radius)
    scipy_normals = []

# -------------寻找法线---------------
# 首先寻找邻域内的点
for i in range(len(scipy_neighbor_idx)):
    neighbor_idx = scipy_neighbor_idx[i] # 得到第i个点的邻近点索引，邻近点包括自己
    neighbor_data = downpcd_np[neighbor_idx] # 得到邻近点，在求邻近法线时没必要归一化，在PCA函数中归一化就行了
    eigenvalues, eigenvectors = PCA(neighbor_data) # 对邻近点做PCA，得到特征值和特征向量
    scipy_normals.append(eigenvectors[:, 2]) # 最小特征值对应的方向就是法线方向
# ------------法线查找结束---------------
scipy_normals = np.array(scipy_normals, dtype=np.float64) # 把法线放在了scipy_normals中


pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(downpcd_np))
pc_view.normals = o3d.utility.Vector3dVector(scipy_normals)
o3d.visualization.draw_geometries([pc_view], "scipy normal estimation", point_show_normal=True)

# 点云法向量的点都以绿色显示
normal_point = o3d.utility.Vector3dVector(scipy_normals)
normals = o3d.geometry.PointCloud()
normals.points = normal_point
normals.paint_uniform_color((0, 1, 0))
o3d.visualization.draw_geometries([downpcd, normals], "scipy noramls points")