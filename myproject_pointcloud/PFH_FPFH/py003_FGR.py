import os
import sys
import copy
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import numpy as np
np.set_printoptions(suppress=True)
import open3d as o3d

# my envs
# env:environment/mmlab_point3d_open3d/bin/python  open3d==0.9.0.0  numpy==1.19.5
# env:environment/miniconda3_py17/bin/python  open3d==0.13.0  numpy==1.21.0

# 以下代码是open3d==0.9.0.0
# 参考
# https://blog.csdn.net/weixin_36219957/article/details/106346469
# http://www.open3d.org/docs/0.12.0/tutorial/pipelines/global_registration.html
# http://www.open3d.org/docs/0.12.0/tutorial/pipelines/global_registration.html#Fast-global-registration
# https://github.com/goncalo120/3DRegNet/blob/master/registration/registration.py
# https://github.com/goncalo120/3DRegNet/blob/master/registration/global_registration.py

voxel_size = 2
source_path = os.path.join(ROOT, "source.ply")
target_path = os.path.join(ROOT, "target.ply")
source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# FPFH
source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

source_fpfh = o3d.registration.compute_fpfh_feature(
    source,
    # o3d.geometry.KDTreeSearchParamHybrid(radius=24, max_nn=256))
    o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel_size, max_nn=100))

target_fpfh = o3d.registration.compute_fpfh_feature(
    target,
    # o3d.geometry.KDTreeSearchParamHybrid(radius=24, max_nn=256))
    o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel_size, max_nn=100))


distance_threshold = 100
result = o3d.registration.registration_fast_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh,
    o3d.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold))
FGR_transformation = result.transformation
FGR_estimate = copy.deepcopy(source)
FGR_estimate.transform(FGR_transformation)

colors = [[1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0],
          [0.0, 1.0, 1.0],
          [1.0, 0.0, 1.0]]
vis_cloud = [
    source, # 红色
    target, # 绿色
    FGR_estimate, # 蓝色
]

for i, pcd in enumerate(vis_cloud):
    color = colors[i]
    pcd.paint_uniform_color(color)
o3d.visualization.draw_geometries(vis_cloud)