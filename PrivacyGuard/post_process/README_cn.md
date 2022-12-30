# 后置处理防御方法

后置处理防御方法采用非侵入式手段来加强模型的安全性，通过对模型输出结果进行安全加固，降低由于模型输出信息导致的隐私泄露。目前主要用来抵御模型推理和模型窃取攻击。

后置处理防御支持如下方法：

- **Rounding**

Rounding方法对模型输出的精度进行裁剪，降低输出的精度（比如只保留2位小数），这样可以一定程度上让攻击效果变差。

- **Labeling**

Labeling方法只输出最终的标签位置标识（indice），最大限度地减少模型输出信息，从而达到保护模型的隐私和安全的目的。

- **TopK**

TopK方法只让模型输出最高的k项，不输出其他项，一定程度上让攻击变得更加困难。

注意：经过后置处理后的模型不能直接用于训练，若有训练需求，需要通过`origin_network()` API获取原网络模型来进行训练。

## 运行例子

```shell
cd PaddleSleeve/PrivacyGuard/post_process
python3 post_processing_demo.py

```

例子训练一个MNIST模型，然后分别用上述策略对模型进行加固。并打印出原始模型输出与加固后的模型输出。

**结果示例1**：

```
origin network output:  Tensor(shape=[1, 10], dtype=float32, place=CPUPlace, stop_gradient=False,
       [[-0.74999607, -5.18590355, -0.21111321,  1.56509531, -1.66427529, -0.12071573, -5.21341133,  8.79216480, -1.47633922,  1.68675601]])
rounding network output (precision = 2):  Tensor(shape=[1, 10], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[-0.75000000, -5.19000006, -0.20999999,  1.57000005, -1.65999997, -0.12000000, -5.21000004,  8.78999996, -1.48000002,  1.69000006]])
label network output (i.e., label indices):  Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=False,
       [7])
topk network output (i.e., top-k (values, indices) pairs):  (Tensor(shape=[1, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
       [[8.79216480, 1.68675601, 1.56509531]]), Tensor(shape=[1, 3], dtype=int64, place=CPUPlace, stop_gradient=False,
       [[7, 9, 3]]))
```
