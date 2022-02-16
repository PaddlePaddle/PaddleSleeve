# 基于Knockoff模型窃取攻击MNIST模型的例子

如下图所示，本例子模拟恶意攻击者通过查询受害者的模型预测接口，来窃取受害者的AI模型。

1. 首先受害者利用自己的数据训练并部署模型$F_A$。
2. 攻击者选择输入数据和模型结构，利用Knockoff模型窃取攻击extract方法窃取到模型$F_A$。
3. 可选地，攻击者输入测试集，利用攻击模块的evaluate方法评估窃取的模型$F_A$的精度，同时也可以选择地输入受害者模型$F_V$，评估其在测试机上的精度作为对比。


<p align="center">
  <img src="../../../docs/images/knockoff_example.png?raw=true" width="700" title="Knockoff Modle extraction Attack Framework"/>
</p>

## 运行例子

首先需要按照说明安装privbox（[安装教程](../../../README_cn.md###安装)），安装成功后，运行：
```shell

python3 knockoff_extraction_with_mnist.py

```

运行结束后, 可以看到程序输出如下，输出被攻击模型和攻击模型的accuracy

```
Victim model's evaluate result:  [xxx]
Knockoff model's evaluate result:  [xxx]
```
例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=128): 训练及预测数据的batch size。
- `--epochs` (int, default=2): 训练被攻击模型和攻击模型的迭代轮数。
- `--learning_rate` (float, default=0.01): 训练被攻击模型和攻击模型的学习率。
- `--num_queries` (int, default=2000): 攻击者的查询次数。
- `--knockoff_net` (str, default="linear"): 攻击者使用的模型结构，可选模型结构为"linear"和"resnet"。
- `--knockoff_dataset` (str, default="mnist"): 训练knockoff模型选用的数据集, 可以是 'mnist' (跟受害者数据集标签100%重叠) 或者 'fmnist' (跟受害者数据集标签0%重叠).
- `--policy` (str, default="random"): 采样策略，即攻击者从数据集中如何采样数据来训练攻击模型。可选为"random"和"adaptive"。"adaptive"策略要求数据必须具有标签。
- `--reward` (str, default="all"): 采样策略为"adaptive"时，有4中激励机制可以选择。"certainty"机制奖励被攻击模型confident高的数据；"diversity"机制奖励关注所有标签的利用率；"loss"机制则会关注攻击模型和被攻击模型之间的loss差异；"all"机制表示上述3种机制的综合。

**结果示例1**：

输入参数：
```shell
batch_size=128, epochs=2, knockoff_net='linear', knockoff_dataset='fmnist', learning_rate=0.01, num_queries=40000, policy='random', reward='all'
```

评估结果：
```
Victim model's evaluate result:  [0.906]
Knockoff model's evaluate result:  [0.862]
```
**结果示例2**：

输入参数：
```shell
batch_size=128, epochs=2, knockoff_dataset='mnist', knockoff_net='linear', learning_rate=0.01, num_queries=6000, policy='adaptive', reward='all'
```

评估结果：
```
Victim model's evaluate result:  [0.8905]
Knockoff model's evaluate result:  [0.886]
```