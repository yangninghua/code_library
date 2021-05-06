# python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: py003_RemoveStatisticalOutliers.py
# @Author: ---
# @Institution: BeiJing, China
# @E-mail: lgdyangninghua@163.com
# @Site: 
# @Time: 4月 24, 2021
# ---
import numpy as np
import open3d as o3d
import math

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("cloud_bin_2.pcd")
#o3d.visualization.draw_geometries([pcd])

def SelectByIndex(PointCloud, indices, invert=False):
    mask = [invert] * len(PointCloud)
    for i in indices:
        mask[i] = ~invert
    output = []
    for ind,val in enumerate(PointCloud):
        if mask[ind]:
            output.append(PointCloud[ind])
    print("Pointcloud down sampled from {} points to {} points.".format(len(PointCloud), len(output)))
    return np.array(output)

def RemoveStatisticalOutliers(PointCloud, nb_neighbors, std_ratio):
    if nb_neighbors <1 or std_ratio<=0:
        print("[RemoveStatisticalOutliers] Illegal input parameters, number ")
        print("of neighbors and standard deviation ratio must be positive")
    if len(PointCloud)==0:
        return np.array([]), []
    temp = o3d.geometry.PointCloud()
    temp.points = o3d.utility.Vector3dVector(PointCloud)
    kdtree = o3d.geometry.KDTreeFlann(temp)
    avg_distances = [0.0] * len(PointCloud)
    indices = []
    valid_distances = 0
    for ind,val in enumerate(PointCloud):
        tmp_indices = []
        dist = []
        # temp = PointCloud
        top_k, tmp_indices, dist = kdtree.search_knn_vector_3d(temp.points[ind], int(nb_neighbors))
        np_dist = np.array(dist)
        mean = -1.0
        if len(dist) > 0:
            valid_distances+=1
            np_dist = np.sqrt(np_dist)
            mean = np.sum(np_dist)/len(np_dist)
        avg_distances[ind] = mean
    if valid_distances ==0:
        return np.array([]), []
    cloud_mean = 0.0
    for p_ind in range(len(avg_distances)):
        if avg_distances[p_ind] > 0:
            cloud_mean += avg_distances[p_ind]
    cloud_mean = cloud_mean/valid_distances
    #https://en.cppreference.com/w/cpp/algorithm/inner_product
    sq_sum = 0
    for m_ind in range(len(avg_distances)-1):
        x = avg_distances[m_ind]
        y = avg_distances[m_ind+1]
        op2 = ((x - cloud_mean) * (y - cloud_mean)) if x > 0 else 0
        sq_sum = sq_sum + op2
    #Bessel's correction
    std_dev = math.sqrt(sq_sum / (valid_distances - 1))
    distance_threshold = cloud_mean + std_ratio * std_dev
    for n_ind, n_val in enumerate(avg_distances):
        if avg_distances[n_ind]>0 and avg_distances[n_ind]<distance_threshold:
            indices.append(n_ind)
    return SelectByIndex(PointCloud, indices), indices

def test_filters_fun(numpy_open3d_result, numpy_python_result):
    row_o3d_Max = np.sum(np.max(numpy_open3d_result, axis=0))
    row_o3d_Min = np.sum(np.min(numpy_open3d_result, axis=0))
    col_o3d_Max = np.sum(np.max(numpy_open3d_result, axis=1))
    col_o3d_Min = np.sum(np.min(numpy_open3d_result, axis=1))

    row_py_Max = np.sum(np.max(numpy_python_result, axis=0))
    row_py_Min = np.sum(np.min(numpy_python_result, axis=0))
    col_py_Max = np.sum(np.max(numpy_python_result, axis=1))
    col_py_Min = np.sum(np.min(numpy_python_result, axis=1))

    print("row_o3d_Max:row_py_Max", row_o3d_Max, row_py_Max)
    print("row_o3d_Min:row_py_Min", row_o3d_Min, row_py_Min)
    print("col_o3d_Max:col_py_Max", col_o3d_Max, col_py_Max)
    print("col_o3d_Min:col_py_Min", col_o3d_Min, col_py_Min)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
np_cl, np_ind = RemoveStatisticalOutliers(np.array(pcd.points), nb_neighbors=20, std_ratio=2.0)
test_filters_fun(np.array(cl.points), np_cl)
