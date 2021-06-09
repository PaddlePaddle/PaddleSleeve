# PrivBox--隐私攻击检测工具

PrivBox是基于PaddlePaddle的AI隐私安全性测试的Python工具库。PrivBox从攻击的角度，提供了多种AI模型隐私攻击的前沿成果的实现，攻击类别包括模型逆向、成员推理、属性推理、模型窃取，帮助模型开发者们通过攻击来更好地评估自己模型的隐私风险。

<p align="center">
  <img src="docs/images/PrivBox.png?raw=true" width="500" title="PrivBox Framework">
</p>


**PrivBox已支持攻击方法**
| 攻击类别  | 已实现方法                                                      |
| -------- | ------------------------------------------------------------ |
| 模型逆向  |                                                                |
|          | Deep Leakage from Gradients（DLG, [论文链接](https://arxiv.org/pdf/1906.08935.pdf)) |
|          | Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning （GAN, [论文链接](https://arxiv.org/pdf/1702.07464.pdf)） |
|          |                                                              |
| 成员推理  |                                                              |
|          |                                                              |
| 属性推理  |                                                              |
|          |                                                              |
| 模型窃取  |                                                              |
|          |                                                              |


## 开始使用


### 环境要求
python 3.6 及以上。

PaddlePaddle 2.0 及以上 (paddle 安装请参考[paddle安装](https://www.paddlepaddle.org.cn/install/quick))。


### 使用例子

使用例子请参考`examples/`目录下对应的[例子目录](examples/)。


## 贡献代码

[PrivBox设计与开发指南](docs/README_cn.md)。

本代码库正在不断开发中，欢迎反馈、报告错误和贡献代码！