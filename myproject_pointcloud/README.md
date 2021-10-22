## 法向量估计

http://www.connellybarnes.com/work/class/2013/shape/04_normal.pptx

https://zhuanlan.zhihu.com/p/56541912

https://www.zhihu.com/people/Adastaybrave/posts



用最直观的方式告诉你：什么是主成分分析PCA

https://www.bilibili.com/video/BV1E5411E71z

【学长小课堂】什么是奇异值分解SVD--SVD如何分解时空矩阵

https://www.bilibili.com/video/BV16A411T7zX

深入浅出了解PCA (Principal Component Analysis)（中英字幕）

https://www.bilibili.com/video/BV1fD4y1X7fw

协方差和相关系数 之 协方差(Covariance and Correlation Part 1 (of 2): Covariance）

https://www.bilibili.com/video/BV1rr4y1T72X

由浅入深告诉你什么是PCA和KPCA

https://www.bilibili.com/video/BV1ex41157bH

http://blog.codinglabs.org/articles/pca-tutorial.html

22降维算法-PCA主成分分析

https://www.bilibili.com/video/BV1PJ411G74g

![](README.assets/04_normal_2.png)

假设初始点云是没有法矢的，我们的目的是为了求取法矢。

在每个点云点 x 分配一个法向量 n

通过拟合局部平面来估计方向

通过传播（生成树）找到一致的全局方向



![04_normal_3](README.assets/04_normal_3.png)

![04_normal_4](README.assets/04_normal_4.png)

![04_normal_5](README.assets/04_normal_5.png)

法矢的特性是**垂直于其点所在的平面**。所谓三点(不共线)确定一个平面，所以我我们可以猜想：采用当前点的邻近点，拟合出一个局部平面，那么法矢就好求了。

![04_normal_6](README.assets/04_normal_6.png)

![04_normal_7](README.assets/04_normal_7.png)

其它出的点也一样：

![04_normal_8](README.assets/04_normal_8.png)

![04_normal_9](README.assets/04_normal_9.png)

![04_normal_10](README.assets/04_normal_10.png)

![04_normal_11](README.assets/04_normal_11.png)

![04_normal_12](README.assets/04_normal_12.png)

![04_normal_13](README.assets/04_normal_13.png)

![04_normal_14](README.assets/04_normal_14.png)

但是会出现二异性问题，毕竟经过一点且垂直一个面的法矢是有两条的：

![04_normal_15](README.assets/04_normal_15.png)

![04_normal_16](README.assets/04_normal_16.png)

所以后续还需要用特定的方法进行法矢定向，这里先不用展开

还记得现在的问题思路么？

**求取某点法矢--》先求该点所在的拟合平面--》求拟合平面？**

好了，理解了问题的出发点，下一步就是寻找解决方案了。

![04_normal_17](README.assets/04_normal_17.png)

拟合出的平面应当具有一个性质：**候选点到这个平面的距离最小**

翻译一下，局部平面拟合的方式：

- 选取当前点的 *k* 个临近点（上述图都 k=2）。或者划定一个半径为 *r* 的球，选取球内部所在的点
- 找到一个平面，使得以上选出得到点到这个平面的距离最小

![04_normal_18](README.assets/04_normal_18.png)

![04_normal_19](README.assets/04_normal_19.png)

（所以你明白为什么pcl做这些的时候为什么要建立 *kd-tree*/octree, 还限定*search radius*了吧）

说到平面拟合，我们最常见的当然是最小二乘法了：

下图首先给出的二维情况下，回顾一下点集拟合直线的情况。

![04_normal_20](README.assets/04_normal_20.png)

![04_normal_21](README.assets/04_normal_21.png)

![04_normal_22](README.assets/04_normal_22.png)

但是我们是要拟合平面是针对法向量，所以这里的距离，只考虑垂直距离：

![04_normal_23](README.assets/04_normal_23.png)

- 好了，PCA终于上场了。忘记大家通常说的PCA的先验知识吧，比如用来降维什么的。就把场景限定在这里：

  - PCA是**为了找到一组新的变换后的正交基（下图绿色部分），这组正交基是给定点集的最佳表达**。

  可以理解为PCA的问题其实是一个正交基的变换，使得变换后的**数据有着最大的方差**

  PCA 找到最能代表给定数据集的正交基

  PCA 找到最佳近似值 线/平面/方向…

![04_normal_24](README.assets/04_normal_24.png)

到这里，你可能尚且不明白它的目的，但是看起来，我们似乎要**建立一个新的正交坐标系统**。

好了，还记得之前的问题场景么？我们是要寻找一个平面，使得点到这个平面的距离最小。

假设这个平面的中心点为 **c** , 穿过中心点的法矢为 **n**, 那么点 **xi** 到平面的距离 **d** ，可以看作是向量 **xi-c** 在法线方向上投影向量的长度，所以这个问题就化解为下图框起来的部分。

补充知识点：向量A在向量B上的投影长度计算：A·B = |A|·|B| cos(θ) = A ^ T*



![04_normal_25](README.assets/04_normal_25.png)

接下来是很多数学的部分！可能你看不懂，但是请记住涉及到的关键点：**协方差矩阵和奇异值分解。**

这是我看PCA的很大一个障碍，我一度不知道为什么要这么做。我会在下一节里去阐述。

定义中心原点：

![04_normal_26](README.assets/04_normal_26.png)

这个m为什么可以是原点，因为原点的特性就是其他点到它的距离的平方和最小。

接下来，问题转化为：

ps：**n**向量是法线，是单位向量，平方和为1

解释一下这个式子：
$$
\mathbf{m}=\underset{\mathbf{c}}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|\mathbf{x}_{i}-\mathbf{c}\right\|^{2}
$$

```text
备注：数学公式里argmin 就是使后面这个式子达到最小值时的变量的取值。也就是，当所有候选点到目标点c距离的平方和最小时，m就是这个c点
```



下面式子中非 二范数 是平方差的意思

m 将是（超）平面的原点

![04_normal_27](README.assets/04_normal_27.png)

到这里问题一路化简为，（不要忘记 **yi** 的含义），下面的式子看着脑壳疼，但是无非就是把积分号移动了，注意 **n** 是未知待求解的。

构建协方差矩阵S：
$$
S=Y Y^{\top}
$$
其中
$$
Y=\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
\mathbf{y}_{1} & \mathbf{y}_{2} & \cdots & \mathbf{y}_{n} \\
\mid & \mid & &
\end{array}\right]
$$
也就是 前文上图中的 **yi**

奇异值分解：
$$
Y=U \Sigma V^{\top}
$$
取U中的最后一列作为法矢



![04_normal_28](README.assets/04_normal_28.png)

![04_normal_29](README.assets/04_normal_29.png)

![04_normal_30](README.assets/04_normal_30.png)

![04_normal_31](README.assets/04_normal_31.png)

![04_normal_32](README.assets/04_normal_32.png)

![04_normal_33](README.assets/04_normal_33.png)

你可能对步骤1和步骤2还能跟上。到了奇异值分解就开始疑问了。不要着急，下面来说为什么要做这样的操作。

协方差矩阵、奇异值分解、特征向量、特征值是为了干嘛？

奇异值分解：
这里比较推荐这篇文章，图文并茂。关键是从几何角度阐释了奇异值分解的意义。

奇异值分解(SVD) --- 几何意义

http://blog.sciencenet.cn/blog-696950-699432.html

http://www.ams.org/publicoutreach/feature-column/fcarc-svd

如果你看这篇文章还是觉得数学很多，那么我定性的给你解释：
$$
Y=U \Sigma V^{\top}
$$
在此处，可以把矩阵 **Y** 看作三个其它的矩阵相乘

- **U** 表示经过变换之后新的坐标系下的正交基
- ![img](README.assets/v2-255278ff5d804ac0241cbfb4caf9d411_720w.png)代表V中的向量与U对应的向量的变换关系
- **V** 代表变换前原坐标系下的正交基坐标系，T代表转置（有些地方写的是*）



![img](README.assets/v2-e1bbab1aa66fae050c65e3f73f947656_720w.jpg)

至于怎么求解，可以看一下这片文章：

https://www.cnblogs.com/pinard/p/6251584.html

奇异值分解(SVD)原理与在降维中的应用

https://www.cnblogs.com/pinard/p/6239403.html

主成分分析（PCA）原理总结

![04_normal_34](README.assets/04_normal_34.png)

![04_normal_35](README.assets/04_normal_35.png)

![04_normal_36](README.assets/04_normal_36.png)

我们看到最后取的法向量就是U中的最后一列，也就是特征值最小的那些特征向量。为什么呢，这就要说到协方差矩阵的意义了。

 **协方差矩阵衡量了沿着特定方向v的点的相关性**

举例：如果找一条穿过原点 ***m*** 的直线 ***l*** , 直线 ***l\*** 沿着方向 ***v*** , 把点xi 投影在直线上，

得到点 ***xi'*** 。 然后下面这一串数学公式，表达的就是点 ***xi'*** 的相关性

![04_normal_37](README.assets/04_normal_37.png)

可以看到相关性越大（对应的特征值也就越大）的点，几乎就快落在（甚至重合）直线l上。

![04_normal_38](README.assets/04_normal_38.png)

![04_normal_39](README.assets/04_normal_39.png)

![04_normal_40](README.assets/04_normal_40.png)

![04_normal_41](README.assets/04_normal_41.png)

![04_normal_42](README.assets/04_normal_42.png)

![04_normal_43](README.assets/04_normal_43.png)

