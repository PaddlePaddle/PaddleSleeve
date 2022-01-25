# 基于规则的成员推理攻击例子

本例子基于简单的规则：预测正确则认为是成员，进行成员攻击，一般会将此种简单方法作为成员攻击的baseline，攻击效果取决于模型的过拟合程度。


## 运行例子

首先需要按照说明安装privbox（[安装教程](../../../README_cn.md###安装)），安装成功后，运行：
```shell

python3 baseline_with_cifar10.py

```

例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=128): 训练目标模型使用数据的batch size。
- `--train_epoch` (int, default=10): 训练目标模型的epoch数量。
- `--train_lr` (float, default=0.0002): 训练目标模型的学习率。

**结果示例1**：

输入参数：
```shell
batch_size=128, train_epoch=10, train_lr=0.0002
```

评估结果：
```shell
Evaluate result of baseline membership attack on cifar10 is: acc: 0.72394,
          auc: 0.7993062947773256, precision: 0.97484， recall: 0.6491143960580636
```