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

# target=31101*6， target_fpfh=33*31101
# FPFH-bin = 3*B = 3*11 = 33
# PFH-bin = B^3 = 5^3 = 125
target_fpfh = o3d.registration.compute_fpfh_feature(
    target,
    # o3d.geometry.KDTreeSearchParamHybrid(radius=24, max_nn=256))
    o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel_size, max_nn=100))

distance_threshold = voxel_size * 1.5

result = o3d.registration.registration_ransac_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh, distance_threshold,
    o3d.registration.TransformationEstimationPointToPoint(False), 3, [
        o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold)
    ], o3d.registration.RANSACConvergenceCriteria(100000, 9999))

FPFH_transformation = result.transformation
FPFH_estimate = copy.deepcopy(source)
FPFH_estimate.transform(FPFH_transformation)


# ICP 以FPFH做初始化
# icp有两种方式o3d.registration.TransformationEstimationPointToPlane()需要法向量
max_iteration=10
max_correspondence_distance=10
estimation_method = o3d.registration.TransformationEstimationPointToPoint()
init_transformation = FPFH_transformation
relative_fitness=1e-6
relative_rmse=1e-6
reg_p2p = o3d.registration.registration_icp(
    source=source,
    target=target,
    init=init_transformation,
    max_correspondence_distance=max_correspondence_distance,
    estimation_method=estimation_method,
    criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration,
                                                        relative_fitness=relative_fitness,
                                                        relative_rmse=relative_rmse),
)
ICP_transformation = reg_p2p.transformation
ICP_estimate = copy.deepcopy(source)
ICP_estimate.transform(ICP_transformation)
metric = dict(
    fitness=reg_p2p.fitness,
    inlier_rmse=reg_p2p.inlier_rmse,
    correspondence_set=reg_p2p.correspondence_set
)


colors = [[1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0],
          [0.0, 1.0, 1.0],
          [1.0, 0.0, 1.0]]
vis_cloud = [
    source, # 红色
    target, # 绿色
    FPFH_estimate, # 蓝色
    ICP_estimate # 黄色
]

for i, pcd in enumerate(vis_cloud):
    color = colors[i]
    pcd.paint_uniform_color(color)
o3d.visualization.draw_geometries(vis_cloud)