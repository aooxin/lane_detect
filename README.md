# BJTU小学期车道线检测作业

final_mark.py是最终作业，前面的都是一部分一部分测试的功能，用的是霍夫变换拟合，测试视频附在百度云中。

[百度云](http:链接：https://pan.baidu.com/s/12NSgBUOF-V26DCJatnmxig)
提取码：hhhh 

下面的是直接截取作业报告里的部分代码，应该疏漏非常多，不过也可以参考叭。

# 关于OpenCV的车道线检测

# 前期芝士

## 1.1基本方法

### 1.1.1图像处理

图像处理主要是先对图像进行灰度处理，高斯模糊，然后对其进行canny边缘检测，最后对得到的图像进行roi掩膜处理，进一步缩小范围。

### 1.1.2霍夫变换

霍夫变换(Hough)是一个检测间断点边界形状的方法。它通过将图像坐标空间变换到参数空间，来实现直线与曲线的拟合。

在图像坐标空间中，经过点的直线表示为：

(1)

其中，参数a为斜率，b为截矩。其中，参数a为斜率，b为截矩。

通过点 点的直线有无数条，且对应于不同的a和b值。

如果将和视为常数，而将原本的参数a和b看作变量，则式子(1)可以表示为：

(2)

这样就变换到了参数平面a−b。这个变换就是直角坐标中对于点的Hough变换。

该直线是图像坐标空间中的点在参数空间的唯一方程。考虑到图像坐标空间中的另一点，它在参数空间中也有相应的一条直线，表示为：

(3)

这条直线与点在参数空间的直线相交于一点，如图所示：

![这里写图片描述](https://pic.imgdb.cn/item/60f934565132923bf8eee8fd.png)

图 3直角变换中的直线霍夫变换

图像坐标空间中过点和点的直线上的每一点在参数空间a−b上各自对应一条直线，这些直线都相交于点,而a0、b0就是图像坐标空间x−y中点和点所确定的直线的参数。

反之，在参数空间相交于同一点的所有直线，在图像坐标空间都有共线的点与之对应。根据这个特性，给定图像坐标空间的一些边缘点，就可以通过Hough变换确定连接这些点的直线方程。

具体计算时，可以将参数空间视为离散的。建立一个二维累加数组,第一维的范围是图像坐标空间中直线斜率的可能范围，第二维的范围是图像坐标空间中直线截矩的可能范围。开始时初始化为0，然后对图像坐标空间的每一个前景点,将参数空间中每一个的离散值代入式子(2)中，从而计算出对应的值。每计算出一对都将对应的数组元素加1，即。所有的计算结束之后，在参数计算表决结果中找到的最大峰值，所对应的、就是源图像中共线点数目最多(共个共线点)的直线方程的参数；接下来可以继续寻找次峰值和第3峰值和第4峰值等等，它们对应于原图中共线点略少一些的直线。

对于上图的Hough变换空间情况如下图所示。

![](https://pic.imgdb.cn/item/60f934485132923bf8eec0b2.png)

图 4直角坐标下的霍夫变换

### 1.1.3离群变换和最小二乘拟合

设置delta值，将不合理的斜率从斜率集中剔除。

利用numpy的最小二乘拟合将所有斜率大于0的点集和斜率小于0的点集分别拟合成两条直线。

### 1.1.4视频流的读写

对视频流逐帧读取并且逐帧处理，最后在进行逐帧播放即可。

## 1.2实验基本流程

1.  Canny边缘检测

2.  手动分割路面区域

3.  霍夫变换得到车道线

4.  获取车道线并叠加到原始图像中

# 局部代码

## 2.1图像处理

### 2.1.1局部二值化处理

要对图像进行边缘检测，首先对图像进行灰度变换，使图像只包含一个通道的信息，然后比较各相邻像素间的亮度差别，亮度产生突变的地方就是边缘像素，将这些边缘像素点连接到一起就形成了边缘图像。

那么首先要知道如何检测出边缘：

边缘有方向和幅值两个要素，通常对图像相邻域像素求取梯度来描述和检测边缘。

在进行边缘检测之前至少要将图像灰度化，因为梯度运算并不能反映色彩的变化差异，所以转换成只有一种颜色通道的灰度图像能够更好地进行边缘检测。

深入了解过图像二值化和边缘检测之后，我认为既可以直接使用灰度图像进行边缘检测，也可以二值化之后再进行边缘检测，二值化的目的是进一步简化灰度图像，使图像中的信息更加纯粹，边缘亮度变化更加明显。如果阈值选的较好还可以滤除不需要的弱边缘，使边缘处理后的图像轮廓更加清晰，效果如图。

![](https://pic.imgdb.cn/item/60f9345c5132923bf8eefba5.png)

图 5局部二值化图像得到的边缘检测图像

![](https://pic.imgdb.cn/item/60f934485132923bf8eec0a4.png)

图 6灰度图的边缘检测图像

可以明显发现对二值化图像进行边缘检测比直接对灰度图进行边缘检测的效果要好，得到的边更宽，可以方便后续操作。

同时，显然局部二值化处理的结果要比全局二值化处理的结果好，效果如图：

![](https://pic.imgdb.cn/item/60f9345c5132923bf8eefc03.png)

![](https://pic.imgdb.cn/item/60f934485132923bf8eec0d5.png)

```
明显全局二值化的图像已经不能看了。代码如下：

def get_bin_img_1(*color_img*):

"""

局部自适应阈值二值化

"""

gray_img = cv2.cvtColor(*color_img*, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(

gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10

)

return binary

def get_bin_img_2(*image*):

"""

全局自适应阈值二值化

"""

gray = cv2.cvtColor(*image*, cv2.COLOR_RGB2GRAY) \# 把输入图像灰度化

ret, binary = cv2.threshold(

gray, 0, 255, cv2.THRESH_BINARY \| cv2.THRESH_TRIANGLE

) # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。

return binary

最后又测试了对边缘处理的图像二值化处理后再计算，效果也还可以，不过似乎差距不大，代码如下：

def test_b_e(*color_img*):

"""先边缘再二值化"""

img = do_do_do(*color_img*)

binary = cv2.adaptiveThreshold(

img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10

)

mask_img_gray = roi_mask(binary)

lines = get_lines(mask_img_gray)

draw_lines(*color_img*, lines)

# return edge_img

return *color_img*

return binary
```

最后权衡之下选择了对边缘处理的图像二值化处理后再计算的这种方法。

### 2.1.2高斯滤波参数选择

经过多次尝试，我选择gaussian_ksize=5, gaussian_sigmax=1，

代码如下：

```
gaussian = cv2.GaussianBlur(

*color_img*, (*gaussian_ksize*, *gaussian_ksize*), *gaussian_sigmax*

)
```

### 2.1.3对图像的roi掩膜处理

对图像进行了较为精确的测算，得出了视频对应的roi范围，效果如图：

```python
def* roi_mask(*gray_img*):

"""

对gray_img进行掩膜

:param gray_img: 灰度图,channels=1

"""

# poly_pts = np.array([[[125, 324], [235, 259], [325, 259], [435, 324]]])
#视频1

poly_pts = np.array([[[118, 243], [293, 112], [365, 110], [575, 243]]]) \#视频2

mask = np.zeros_like(*gray_img*)

mask = cv2.fillPoly(mask, *pts*=poly_pts, *color*=255)

img_mask = cv2.bitwise_and(*gray_img*, mask)

return img_mask
```



## 2.2车道线计算

### 2.2.1剔除离群点

利用斜率的合理性，剔除误差较大的点，代码如下

```python
def* reject_abnormal_lines(*lines*, *threshold*=0.2):

"""

剔除不一致的线段

"""

slopes = [calculate_slope(line) for line in *lines*]

while len(*lines*) \> 0:

mean = np.mean(slopes)

diff = [abs(s - mean) for s in slopes]

idx = np.argmax(diff)

if diff[idx] \> *threshold*:

slopes.pop(idx)

*lines*.pop(idx)

else:

break

return *lines
```

### 合理性判断

在绘制直线之前，对将要绘制的直线进行合理性判断，因为车道线不会突变，所以主要是利用上一帧的斜率以及大致斜率来排除，代码如下：

*

```python
def* draw_lines(*img*, *lines*):

"""

绘制线段

"""

try:

x = False

global pre_lines

left_line, right_line = *lines*

\# print(left_line)

\# print((left_line[0][1] - left_line[1][1]) / (left_line[0][0] -
left_line[1][0]))

if (

(left_line[0][1] - left_line[1][1]) / (left_line[0][0] - left_line[1][0])

) \> 0.5:

"""print(

(left_line[0][1] - left_line[1][1])

/ (left_line[0][0] - left_line[1][0])

)"""

cv2.line(

img,

tuple(left_line[0]),

tuple(left_line[1]),

color*=(0, 255, 255),

thickness*=4,

)

x = True

else:

left_line, right_line = pre_lines

if (

(left_line[0][1] - left_line[1][1])

/ (left_line[0][0] - left_line[1][0])

) > 0.5:

cv2.line(

*img*,

tuple(left_line[0]),

tuple(left_line[1]),

*color*=(0, 255, 255),

*thickness*=4,

)

x = False

# print(right_line)

if (

(right_line[0][1] - right_line[1][1])

/ (right_line[0][0] - right_line[1][0])

) < -0.5:

cv2.line(

*img*,

tuple(right_line[0]),

tuple(right_line[1]),

*color*=(0, 255, 255),

*thickness*=4,

)

x = True

else:

left_line, right_line = pre_lines

if (

(right_line[0][1] - right_line[1][1])

 (right_line[0][0] - right_line[1][0])

) < -0.5:

cv2.line(

img,

tuple(right_line[0]),

tuple(right_line[1]),

color=(0, 255, 255),

thickness*=4,

)

x = False

if x == True:

pre_lines = lines

except BaseException:

return
```

