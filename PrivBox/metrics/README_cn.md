# 评估指标
English | [简体中文](./README_cn.md)


目录包含多种用来评估攻击效果方法，这些方法包括：准确率（Accuracy）、召回率（Recall）、精确率（Precision）、AUC（Area under the curve）、均方误差（Mean squared error, MSE）、峰值信噪比（Peak signal-to-noise ratio, PSNR）和结构相似性（Structural similarity, SSIM）。

## 准确率、精确率、召回率和AUC
设:
```
TP：正例预测正确的个数
FP：负例预测错误的个数
TN：负例预测正确的个数
FN：正例预测错误的个数
```

各评估指标计算如下

- 准确率
  
  $$ACC = (TP + TN) / (TP + TN + FP + FN)$$

- 精确率
  
$$PRECISION = TP / (TP + FP)$$

- 召回率
  
  $$RECALL = TP / (TP + FN)$$

- AUC

计算AUC还需要两个指标：

所有正确预测中正例的比例：$TPR = TP / (TP + FN)$

所有错误预测中负例的比例：$FPR = FP / (FP + TN)$
显然，随着分类门限值的不同，就会产生很多的$TPR$和$FPR$值。我们以FPR为横坐标，TPR为纵坐标可绘制出一条曲线，称为ROC曲线，该曲线的与横坐标覆盖的面积记为AUC值，值越接近1表示分类效果越好（详细的AUC定义参考[链接](https://en.wikipedia.org/w/index.php?title=Area_under_the_curve_(receiver_operating_characteristic)&redirect=no)）。

## 均方误差

均方误差表示预测标签与实际标签之间的误差大小，误差越小表示预测结果与实际结果越接近。计算公式为

$$MSE = (Y - \hat{Y})^2 / n$$
$n$为样本数量。

## 峰值信噪比

峰值信噪比用来衡量两个图像的差异，值越高表示两个图像差异越小。计算公式如下：

$$PSNR = 20 * log10(MAX / \sqrt{MSE})$$
$MSE$ 为均方误差， $MAX$为图像像素的最大值。

## 结构相似性

结构相似性是用来衡量两张图像的相似程度指标，相比于峰值信噪比指标，结构相似性指标更符合人眼对图像相似性的判断。值越接近1代表图像越相似。

SSIM计算会先将图片划分为多个窗口，对每个窗口分别计算SSIM， 计算公式如下：

$$SSIM(x, y) = (2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2) / ((\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2))$$

其中，$\mu_x$为$x$的均值，$\mu_y$为$y$的均值，$\sigma_x^2$为$x$的方差，$\sigma_y^2$是$y$ 的方差，$\sigma_{xy}$是$x$和$y$的协方差，$c_1 = (k_1L)^2, ~c_2=(k_2L)^2$用来稳定图像的亮度、对比度、结构，$L$为像素的最大值，$k_1 = 0.01, ~k_2 = 0.03$为常数.

更多说明参考[链接](https://en.wikipedia.org/wiki/Structural_similarity)