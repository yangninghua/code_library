# python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: py001_UniformDownSample.py
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

def UniformDownSample(PointCloud, every_k_points):
    if every_k_points == 0:
        print("[UniformDownSample] Illegal sample rate.")
    indices = []
    rows = len(PointCloud)
    for ind in range(0, rows, every_k_points):
        indices.append(ind)
    return SelectByIndex(PointCloud, indices)

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

uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
np_uni_down_pcd = UniformDownSample(np.array(pcd.points), 5)
test_filters_fun(np.array(uni_down_pcd.points), np_uni_down_pcd)