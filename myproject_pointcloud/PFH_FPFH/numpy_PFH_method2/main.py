#!/opt/conda/envs/feature-detection/bin/python

# main.py
#     1. load point cloud in modelnet40 normal format
#     2. calculate ISS keypoints
#     3. calculate FPFH or SHOT for detected keypoints
#     3. visualize the results

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

import argparse

# IO utils:
from myio import read_modelnet40_normal
# detector:
from iss import detect
# descriptor:
from fpfh import describe

import numpy as np
import pandas as pd
import open3d as o3d

import seaborn as sns
import matplotlib.pyplot as plt

def get_arguments():
    """ 
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Detect ISS keypoints on ModelNet40 dataset.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # -i /workspace/data/modelnet40_normal_resampled/chair/chair_0001.txt -r 0.05

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of ModelNet40 sample.", 
        default="/home/ninghua/windows10/media_2.0T/dataset/ModelNet/modelnet40_normal_resampled/chair/chair_0001.txt",
        required=False
    )
    required.add_argument(
        "-r", dest="radius", help="Radius for radius nearest neighbor definition.",
        required=False, type=float, default=0.05
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    # load point cloud:
    point_cloud = read_modelnet40_normal(arguments.input)
    # compute surface normals:
    # point_cloud.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    # )
    # build search tree:
    search_tree = o3d.geometry.KDTreeFlann(point_cloud)

    # detect keypoints:
    keypoints = detect(point_cloud, search_tree, arguments.radius)
    
    # visualize:

    # paint background as grey:
    point_cloud.paint_uniform_color([0.50, 0.50, 0.50])
    # show roi:
    max_bound = point_cloud.get_max_bound()
    min_bound = point_cloud.get_min_bound()
    center = (min_bound + max_bound) / 2.0

    min_bound[1] = max_bound[1] - 0.1
    max_bound[1] = max_bound[1]
    min_bound[2] = center[2]
    max_bound[2] = max_bound[2]

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound = min_bound,
        max_bound = max_bound
    )

    roi = point_cloud.crop(bounding_box)
    roi.paint_uniform_color([1.00, 0.00, 0.00])

    # paint keypoints as red:
    keypoints_in_roi = keypoints.loc[
        (
            ((keypoints['x'] >= min_bound[0]) & (keypoints['x'] <= max_bound[0])) &
            ((keypoints['y'] >= min_bound[1]) & (keypoints['y'] <= max_bound[1])) &
            ((keypoints['z'] >= min_bound[2]) & (keypoints['z'] <= max_bound[2]))
        ), 
        :
    ]
    np.asarray(point_cloud.colors)[keypoints_in_roi['id'].values, :] = [1.0, 0.0, 0.0]

    o3d.visualization.draw_geometries([point_cloud])

    # describe keypoints:
    df_signature_visualization = []
    for keypoint_id in keypoints_in_roi['id'].values:
        signature = describe(point_cloud, search_tree, keypoint_id, arguments.radius, 6)
        df_ = pd.DataFrame.from_dict(
            {
                'index': np.arange(len(signature)),
                'feature': signature
            }
        )
        df_['keypoint_id'] = keypoint_id # f'{keypoint_id:06d}'

        df_signature_visualization.append(df_)

    # https://discuss.analyticsvidhya.com/t/python-error-cannot-reindex-from-a-duplicate-axis/6333/2
    #df_signature_visualization = pd.concat(df_signature_visualization[:6], ignore_index=True)
    df_signature_visualization = pd.concat(df_signature_visualization, ignore_index=True)
    #print(df_signature_visualization)

    # limit the number:
    #df_signature_visualization = df_signature_visualization.head(5)

    # draw the plot:
    plt.figure(num=None, figsize=(16, 9))

    sns.lineplot(
        x="index", y="feature",
        hue="keypoint_id", style="keypoint_id",
        markers=True, dashes=False, data=df_signature_visualization
    )

    plt.title('Description Visualization for Keypoints')
    plt.show()

