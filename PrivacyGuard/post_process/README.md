# Post-Processing Defenses
English | [简体中文](./README_cn.md)

The post-processing defense method uses non-intrusive method to enhance the security of the model, and reduces the privacy disclosure caused by the model output. It is mainly used to defend against model inference and model extraction attacks.

Post-processing defenses support the following approach:

- **Rounding**

Rounding defense strategy truncates the precision of model output, such as only preserves 2 decimals, which can make attack less effective.

- **Labeling**

The Labeling method only outputs the final label indices, minimizing model output information for the purpose of protecting the privacy and security of the model.

- **TopK**

The TopK method only allows the model to output the highest k-items, not other items, making the attack more difficult.

Notice: Post-processing models cannot be used directly for training, and if training is required, the original network model needs to be obtained through the 'origin_network()' API for training.

## Run Example

```shell
cd PaddleSleeve/PrivacyGuard/post_process
python3 post_processing_demo.py

```

This example trains a MNIST model and then enhances the model with the above strategies separately. The original model output and the post-processing model output are printed as follows.


**Example Result 1**:

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
