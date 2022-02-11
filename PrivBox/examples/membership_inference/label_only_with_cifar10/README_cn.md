# Label-only成员推理攻击例子

本例子使用label-only成员推理攻击模块对目标模型进行攻击，例子的攻击流程如下图所示。

<p align="center">
  <img src="../../../docs/images/labelonly_example.png?raw=true" width="500" title="Label-only membershi attack example"/>
</p>

## 运行例子

```shell

python3 label_only_with_cifar10.py

```

例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=128): 训练模型使用数据的batch size。
- `--target_epoch` (int, default=10): 训练目标模型的epoch数量。
- `--shadow_epoch` (int, default=10): 训练影子模型的epoch数量。
- `--classifier_epoch` (int, default=10): 训练分类器的epoch数量。
- `--target_lr` (float, default=0.0002): 训练目标模型的学习率。
- `--shadow_lr` (float, default=0.0002): 训练影子模型的学习率。
- `--classifier_lr` (float, default=0.0002): 训练分类器的学习率。
- `--shadow_dataset` (str, default=cifar10): 影子数据集，可选cifar10。
- `--target_dataset` (str, default=cifar10): 目标数据集，可选cifar10。
- `--shadow_model` (str, default=resnet18): 影子模型的结构，可选resnet18、resnet34。
- `--attack_type` (str, default=r): 攻击类型，r代表对图像进行旋转。
- `--r` (int, default=6): 指定产生的增强图像个数的参数。


**Example Result 1**：

Input Parameters：

```shell
attack_type='r', batch_size=128, classifier_epoch=10, classifier_lr=0.0002, r=6, shadow_dataset='cifar10', shadow_epoch=10, shadow_lr=0.0002, shadow_model='resnet18', target_dataset='cifar10', target_epoch=10, target_lr=0.0002
```

Evaluation Result：
```shell
Evaluate result of Label-only membership attack is: acc: 0.8333333333333334,
          auc: 0.49814829843749997, precision: 0.8333333333333334， recall: 1.0
```