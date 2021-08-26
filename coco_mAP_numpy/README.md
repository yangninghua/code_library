# coco_mAP_numpy

### 参考
https://cocodataset.org/#detection-eval
https://github.com/rafaelpadilla/review_object_detection_metrics

### 任务启动
python main_inference.py

### 用一个任务与mmdet_v2.8测试对齐效果(4类的检测任务)

```python
mmdet_v2.8的方法
Evaluate annotation type *bbox*
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.897
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.993
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.989
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.717
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.878
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.917
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.930
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.930
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.930
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.763
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.909
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.946
OrderedDict([('bbox_mAP', 0.897), ('bbox_mAP_50', 0.993), ('bbox_mAP_75', 0.989), 
('bbox_mAP_s', 0.717), ('bbox_mAP_m', 0.878), ('bbox_mAP_l', 0.917), 
('bbox_mAP_copypaste', '0.897 0.993 0.989 0.717 0.878 0.917')
])
```

```python
我们的方法
{'AP': 0.8966142541344695,
 'AP50': 0.9925742574257426,
 'AP75': 0.9889083417811588,
 'APlarge': 0.9173113624926404,
 'APmedium': 0.8784283010853097,
 'APsmall': 0.7167888030813641,
 'AR1': 0.2207999991904892,
 'AR10': 0.8166207586245783,
 'AR100': 0.9303258214488423,
 'ARlarge': 0.94639358650855,
 'ARmedium': 0.9091566178490108,
 'ARsmall': 0.7627151051625239}
```