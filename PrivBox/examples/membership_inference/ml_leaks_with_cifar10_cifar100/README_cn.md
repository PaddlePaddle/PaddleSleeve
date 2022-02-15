# ML-Leaks成员推理攻击例子

本例子使用ML-Leaks成员推理攻击模块来对目标模型进行攻击，例子的攻击流程如下图所示。

<p align="center">
  <img src="../../../docs/images/ml_leak_example.png?raw=true" width="500" title="ML-Leaks membershi attack example"/>
</p>

## 运行例子

首先需要按照说明安装privbox（[安装教程](../../../README_cn.md###安装)），安装成功后，运行：
```shell

python3 ml_leaks_with_cifar10_cifar100.py

```

例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=128): 训练模型使用数据的batch size。
- `--target_epoch` (int, default=10): 训练目标模型的epoch数量。
- `--shadow_epoch` (int, default=10): 训练影子模型的epoch数量。
- `--classifier_epoch` (int, default=10): 训练ML-Leaks分类器的epoch数量。
- `--target_lr` (float, default=0.0002): 训练目标模型的学习率。
- `--shadow_lr` (float, default=0.0002): 训练影子模型的学习率。
- `--classifier_lr` (float, default=0.0002): 训练ML-Leaks分类器的学习率。
- `--topk` (int, default=10): 使用top k个预测结果来训练ML-Leaks分类器。
- `--shadow_dataset` (str, default=cifar10): 影子数据集，可选cifar10、cifar100。
- `--target_dataset` (str, default=cifar10): 目标数据集，可选cifar10、cifar100。
- `--shadow_model` (str, default=resnet18): 影子模型的结构，可选resnet18、resnet34。


**结果示例1**：

输入参数：
```shell
batch_size=128, classifier_epoch=10, classifier_lr=0.0002, shadow_dataset='cifar10', shadow_epoch=10, shadow_lr=0.0002, shadow_model='resnet18', target_dataset='cifar10', target_epoch=10, target_lr=0.0002, topk=10
```

评估结果：
```shell
Evaluate result of ML-Leaks membership attack is: acc: 0.7527333333333334,
          auc: 0.733478836, precision: 0.9001112842198976， recall: 0.80884
```