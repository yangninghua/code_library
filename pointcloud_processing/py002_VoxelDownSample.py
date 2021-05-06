# python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: py002_VoxelDownSample.py
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

def GetMinBound(cloud):
    out = np.min(cloud, axis=0)
    return out

def GetMaxBound(cloud):
    out = np.max(cloud, axis=0)
    return out

#open3d - voxel_down_sample
def VoxelDownSample(PointCloud, voxel_size):
    #参考：https://www.cnblogs.com/huashanqingzhu/p/6603875.html
    INT_MAX = 2147483647
    INT_MIN = -2147483648

    output = []
    if voxel_size <= 0.0:
        print("[VoxelDownSample] voxel_size <= 0.")
    voxel_size3 = np.array([voxel_size, voxel_size, voxel_size])
    voxel_min_bound = GetMinBound(PointCloud) - voxel_size3 * 0.5
    voxel_max_bound = GetMaxBound(PointCloud) + voxel_size3 * 0.5

    if  voxel_size*INT_MAX < np.max(voxel_max_bound-voxel_min_bound):
        print("[VoxelDownSample] voxel_size is too small.")

    voxelindex_to_accpoint = {}
    voxelindex_to_accpoint_Average = {}
    for ind,val in enumerate(PointCloud):
        ref_coord = (PointCloud[ind] - voxel_min_bound)/voxel_size
        x,y,z = int(math.floor(ref_coord[0])), int(math.floor(ref_coord[1])), int(math.floor(ref_coord[2]))
        #voxel_index = np.array([x,y,z])
        voxel_index = (x,y,z)
        if voxel_index in voxelindex_to_accpoint:
            voxelindex_to_accpoint[voxel_index] = voxelindex_to_accpoint[voxel_index] + val
            voxelindex_to_accpoint_Average[voxel_index] = voxelindex_to_accpoint_Average[voxel_index] + 1
        else:
            voxelindex_to_accpoint[voxel_index] = val
            voxelindex_to_accpoint_Average[voxel_index] = 1

    for ind,val in enumerate(voxelindex_to_accpoint.keys()):
        output.append(voxelindex_to_accpoint[val]/voxelindex_to_accpoint_Average[val])

    print("Pointcloud down sampled from {} points to {} points.".format(len(PointCloud), len(output)))
    return np.array(output)

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

voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
np_voxel_down_pcd = VoxelDownSample(np.array(pcd.points), voxel_size=0.02)
test_filters_fun(np.array(voxel_down_pcd.points), np_voxel_down_pcd)