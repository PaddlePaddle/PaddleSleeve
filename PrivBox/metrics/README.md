# Metrics
English | [简体中文](./README_cn.md)

There are some metrics implemented in the directory, they are Accuracy、Recall、Precision、AUC (Area under the curve)、MSE (Mean squared error)、PSNR (Peak signal-to-noise ratio) and SSIM (Structural similarity).

## Accuracy、Recall、Precision and AUC
Let:
```
TP: True positive
FP: False positive
TN: True negative
FN: False negative
```

The metrics are computed as
- Accuracy

$$ACC = (TP + TN) / (TP + TN + FP + FN)$$

- Precision
  
$$PRECISION = TP / (TP + FP)$$

- Recall

$$RECALL = TP / (TP + FN)$$

- AUC

For AUC, two additional metric is need:

True positive rate：$TPR = TP / (TP + FN)$

False positive：$FPR = FP / (FP + TN)$

Obliviously, with various threshold setting, we can plot the TPR curve against FPR. The curve is call ROC, and the area under the curve is AUC value. AUC values closer to 1 represent better classification performence. (details see the [link](https://en.wikipedia.org/wiki/Receiver_operating_characteristic))


## MSE (Mean squared error)

MSE measures the average of squared of the errors between expect labels and predicted labels. It is computed as: 

$$MSE = (Y - \hat{Y})^2 / n$$

$n$ is the number of samples.

## PSNR (Peak signal-to-noise ratio)

PSNR measures the difference between two images, it is computed as:

$$PSNR = 20 * log10(MAX / \sqrt{MSE})$$

$MSE$ means squared error， $MAX$ is max value for image pixels.

## SSIM (Structural similarity)

SSIM is used for measuring the similarity between two images. It is computed on various windows of images. For each window, SSIM value is calculated as:

$$SSIM(x, y) = (2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2) / ((\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2))$$
$\mu_x$ is the average of $x$, $\mu_y$ is the average of $y$, $\sigma_x^2$ is the variance of $x$, $\sigma_y^2$ is the variance of $y$, $\sigma_{xy}$ is the covariance of $x$ and $y$, $c_1 = (k_1L)^2, ~c_2=(k_2L)^2$ are two variables to stabilize the division with weak denominator, $L$ is the max value of the image pixel, $k_1 = 0.01, ~k_2 = 0.03$ by default.

SSIM value closer to 1.0 means the two images are more similar. more details see the [link](https://en.wikipedia.org/wiki/Structural_similarity).