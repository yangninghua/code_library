import os
os.environ["CUDA_VISIBLE_DEVICES"]='1' 
import sys
import cv2
import json
import mmcv
from math import isclose
import numpy as np
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.coco import maskUtils
from pycocotools.cocoeval import COCOeval
 

from bounding_box import *
from coco_evaluator import *


def get_file_name_only(file_path):
    if file_path is None:
        return ''
    return os.path.splitext(os.path.basename(file_path))[0]

'''
MSCOCO2014数据集：
训练集： 82783张，13.5GB， 验证集： 40504张，6.6GB，共计123287张
MSCOCO2017数据集：
训练集：118287张，19.3GB， 验证集： 5000张，1814.7M，共计123287张
参考：
https://cocodataset.org/#detection-eval
https://github.com/rafaelpadilla/review_object_detection_metrics
'''

config_file = 'coco_mAP_from_mmdet2.8/chechkpoints/faster_rcnn_r101_fpn_1x_coco.py'
checkpoint_file = 'coco_mAP_from_mmdet2.8/chechkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
model = init_detector(config_file, checkpoint_file, device='cuda')


coco_dataset_path = "dataset/COCO数据集/annotations/instances_val2014.json"
coco_dataset = "dataset/COCO数据集/"
coco = COCO(coco_dataset_path)
catIds = coco.getCatIds(catNms=[], supNms=[], catIds=[])
imgIds = coco.getImgIds(catIds=[])


classes_ori = coco.dataset['categories']
# into dictionary
classes = {c['id']: c['name'] for c in classes_ori}
det_classes = {ind: c['id'] for ind, c in enumerate(classes_ori)}

gts = []
dts = []
count = len(imgIds)
num = 0
for m_ind in range(len(imgIds)):
    num += 1

    print(num, "/", count)

    img_info = coco.loadImgs(imgIds[m_ind])[0]

    cvImage = cv2.imread(os.path.join(coco_dataset, "val2014", img_info['file_name']), -1)

    annIds = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(annIds)

    img_size = (int(img_info['width']), int(img_info['height']))
    img_name = get_file_name_only(img_info['file_name'])
    for index in range(len(anns)):
        annotation = anns[index]
        category_id = annotation['category_id']
        x1, y1, bb_width, bb_height = annotation['bbox']
        img_id = annotation['image_id']

        gts_bb = BoundingBox(image_name=img_name,
                    class_id=classes[annotation['category_id']],
                    coordinates=(x1, y1, bb_width, bb_height),
                    type_coordinates=CoordinatesType.ABSOLUTE,
                    img_size=img_size,
                    confidence=None,
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH)
        gts.append(gts_bb)
    

    result = inference_detector(model, cvImage)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    for n_ind, n_val in enumerate(bbox_result):
        if n_val.size > 0:
            class_id = classes[det_classes[n_ind]]
            for bb_num in range(len(n_val)):
                temp = (n_val[bb_num, :]).tolist()
                x1, y1, x2, y2, confidence = temp
                dts_bb = BoundingBox(image_name=img_name,
                            class_id=class_id,
                            coordinates=(x1, y1, x2, y2),
                            type_coordinates=CoordinatesType.ABSOLUTE,
                            img_size=img_size,
                            confidence=confidence,
                            bb_type=BBType.DETECTED,
                            format=BBFormat.XYX2Y2)
                dts.append(dts_bb)

# COCO评估指标
res = get_coco_summary(gts, dts)
import pprint
pprint.pprint(res)


# VOC评估指标
from pascal_voc_evaluator import get_pascalvoc_metrics, plot_precision_recall_curves
from enumerators import MethodAveragePrecision
testing_ious = [0.1, 0.3, 0.5, 0.75]
# ELEVEN_POINT_INTERPOLATION
for idx, iou in enumerate(testing_ious):
    results_dict = get_pascalvoc_metrics(
        gts, dts, iou_threshold=iou, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
    results = results_dict['per_class']

# EVERY_POINT_INTERPOLATION
for idx, iou in enumerate(testing_ious):
    results_dict = get_pascalvoc_metrics(
        gts, dts, iou_threshold=iou, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)
    results = results_dict['per_class']


