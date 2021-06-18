# 基于规则的成员推理攻击例子

本例子基于简单的规则：预测正确则认为是成员，进行成员攻击，一般会将此种简单方法作为成员攻击的baseline，攻击效果取决于模型的过拟合程度。


## 运行例子

```shell

python3 baseline_with_cifar10.py

```

例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=128): 训练目标模型使用数据的batch size。
- `--train_epoch` (int, default=10): 训练目标模型的epoch数量。
- `--train_lr` (float, default=0.0002): 训练目标模型的学习率。
