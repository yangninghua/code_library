#https://github.com/patvarilly/CoverTree
#https://github.com/Vectorized/Python-KD-Tree
#https://github.com/qzq2514/KDTree_BallTree
#https://github.com/tushushu/imylu
#https://zhuanlan.zhihu.com/p/45346117


import numpy as np
import pandas as pd
from collections import Counter
import time
allow_duplicate = False
class KDNode():
    def __init__(self,value,label,left,right,depth):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
        self.depth = depth

class KDTree():
    def __init__(self,values,labels):
        self.values = values
        self.labels = labels
        if(len(self.values) == 0 ):
            raise Exception('Data For KD-Tree Must Be Not empty.')
        self.dims_len = len(self.values[0])
        self.root = self.build_KDTree()
        self.KNN_result = []
        self.nums=0

    def build_KDTree(self):
        data = np.column_stack((self.values,self.labels))
        return self.build_KDTree_core(data,0)

    def dist(self,point1,point2):
        return np.sqrt(np.sum((point1-point2)**2))

    #data:带标签的数据且已经排好序的
    def build_KDTree_core(self,data,depth):
        if len(data)==0:
            return None
        cuttint_dim = depth % self.dims_len

        data = data[data[:, cuttint_dim].argsort()]  # 按照第当前维度排序
        mid_index = len(data)//2
        node = KDNode(data[mid_index,:-1],data[mid_index,-1],None,None,depth)
        node.left = self.build_KDTree_core(data[0:mid_index],depth+1)
        node.right = self.build_KDTree_core(data[mid_index+1:], depth + 1)
        return node

    def search_KNN(self,target,K):
        if self.root is None:
            raise Exception('KD-Tree Must Be Not empty.')
        if K > len(self.values):
            raise ValueError("K in KNN Must Be Greater Than Lenght of data")
        if len(target) !=len(self.root.value):
            raise ValueError("Target Must Has Same Dimension With Data")
        self.KNN_result = []
        self.nums = 0
        self.search_KNN_core(self.root,target,K)
        return self.nums


    def search_KNN_core(self,root, target, K):
        if root is None:
            return
        cur_data = root.value
        label = root.label
        self.nums+=1
        distance = self.dist(cur_data,target)

        is_duplicate = [self.dist(cur_data, item[0].value)< 1e-4 and
                        abs(label-item[0].label) < 1e-4 for item in self.KNN_result]
        if not np.array(is_duplicate, bool).any() or allow_duplicate:
            if len(self.KNN_result) < K:
                # 向结果中插入新元素
                self.KNN_result.append((root,distance))
            elif distance < self.KNN_result[0][1]:
                # 替换结果中距离最大元素
                self.KNN_result = self.KNN_result[1:]+[(root,distance)]
        self.KNN_result=sorted(self.KNN_result,key=lambda x:-x[1])
        cuttint_dim = root.depth % self.dims_len
        if abs(target[cuttint_dim] - cur_data[cuttint_dim]) < self.KNN_result[0][1] or len(self.KNN_result) < K:
            # 在当前切分维度上,以target为中心,最近距离为半径的超体小球如果和该维度上的超平面有交集,那么说明可能还存在更近的数据点
            # 同时如果还没找满K个点，也要继续寻找(这种有选择的比较,正是使用KD树进行KNN的优化之处,不用像一般KNN一样在整个数据集遍历)
            self.search_KNN_core(root.left,target,K)
            self.search_KNN_core(root.right,target,K)
        # 在当前划分维度上,数据点小于超平面,那么久在左子树继续找,否则在右子树继续找
        elif target[cuttint_dim] < cur_data[cuttint_dim]:
            self.search_KNN_core(root.left,target,K)
        else:
            self.search_KNN_core(root.right,target,K)

import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import numpy as np
np.set_printoptions(suppress=True)
import open3d as o3d

def PCA(data, sort=True):
    # normalize 归一化
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    # 计算对称的协方差矩阵
    H = np.dot(normal_data.T, normal_data)
    #ps在下面的代码中，协方差矩阵除以(n-1)
    #https://github.com/tushushu/imylu/blob/master/imylu/decomposition/pca.py
    # pca给定要降低到的维度ndim，取特征值top-ndim大的对应的特征向量即为pca所求
    # 而我们求的法向量是最小特征值对应的特征向量

    # SVD奇异值分解，得到H矩阵的特征值和特征向量
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)
    if sort:
        #从大到小排序特征值
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

path = os.path.join(ROOT, "bun000.ply")
normalPath = path.replace("bun000.ply", "bun000_normal.ply")
pcd = o3d.io.read_point_cloud(path)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
downpcd = pcd.voxel_down_sample(voxel_size=0.004)
print(downpcd)
downpcd_np = np.array(downpcd.points)

label = np.ones(downpcd_np.shape[:1], np.int8)
kd_tree = KDTree(downpcd_np, label)

scipy_normals = []
for index, target in enumerate(downpcd_np):
    if index % 100 ==0:
        print(index)
    calu_dist_nums = kd_tree.search_KNN(target, 20)
    neighbor_data = [tuple(node[0].value) for node in kd_tree.KNN_result]
    neighbor_data = np.array(neighbor_data)
    # 得到邻近点，在求邻近法线时没必要归一化，在PCA函数中归一化就行了
    eigenvalues, eigenvectors = PCA(neighbor_data) # 对邻近点做PCA，得到特征值和特征向量
    scipy_normals.append(eigenvectors[:, 2]) # 最小特征值对应的方向就是法线方向
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