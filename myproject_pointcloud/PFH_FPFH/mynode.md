numpy实现参考
深蓝学院课程

https://github.com/lijx10/NN-Trees

https://github.com/BayRanger/PointCloudProcessing/tree/master/HW9_new/tools

https://github.com/AlexGeControl/3D-Point-Cloud-Analytics/tree/master/workspace/assignments/08-feature-description

https://github.com/ZC413074/course_of_3d_points/tree/master/lesson8_description

https://github.com/stevenliu216/Point-Feature-Histogram


'''
ISS 点云特征点的提取
PFH FPFH 特征点的描述
两大类
基于直方图 PFH FPFH
基于标记的方法 SHOT

PFH
输入是xyz   法向量  关键点
输出是B^3的数组
思想 旋转3自由度平移3自由度  满足6D-pose不变性
通过周围点法向量变化描述周围的一片区域 

怎么解决旋转平移不变性
每次只考虑两个点之间的关系，关键点与近邻点，近邻点与近邻点
通过两个点及其法向量，建立坐标系，该坐标系只和两个点相关

p1 p2两个点
p1的三个轴分别是
u = p1法向量n1
v = u 叉乘 p1与p2向量的归一化
w = u 叉乘 v
坐标系包含了相对位置信息和法向量信息
'''
