# PrivBox--隐私攻击检测工具

PrivBox是基于PaddlePaddle的AI隐私安全性测试的Python工具库。PrivBox从攻击的角度，提供了多种AI模型隐私攻击的前沿成果的实现，攻击类别包括模型逆向、成员推理、属性推理、模型窃取，帮助模型开发者们通过攻击来更好地评估自己模型的隐私风险。

<p align="center">
  <img src="docs/images/PrivBox.png?raw=true" width="500" title="PrivBox Framework">
</p>


**PrivBox已支持攻击方法**
<table>
   <tr>
      <td>攻击类别</td>
      <td>已实现方法</td>
      <td>参考文献</td>
   </tr>
   <tr>
      <td rowspan="2">模型逆向</td>
      <td>Deep Leakage from Gradients（DLG)</td>
      <td><a href="https://arxiv.org/pdf/1906.08935.pdf">[ZLH19]</a></td>
   </tr>
   <tr>
      <td>Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning (GAN)</td>
      <td><a href="https://arxiv.org/pdf/1702.07464.pdf">[HAPC17]</a></td>
   </tr>
   <tr>
      <td rowspan="2">成员推理</td>
      <td>Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting</td>
      <td><a href="https://arxiv.org/pdf/1709.01604.pdf">[YGFJ18]</a></td>
   </tr>
   <tr>
      <td>ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models</td>
      <td><a href="https://arxiv.org/pdf/1806.01246.pdf">[SZHB19]</a></td>
   </tr>
   <tr>
      <td rowspan="1">属性推理</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td rowspan="1">模型窃取</td>
      <td></td>
      <td></td>
   </tr>
</table>


## 开始使用


### 环境要求
python 3.6 及以上。

PaddlePaddle 2.0 及以上 (paddle 安装请参考[paddle安装](https://www.paddlepaddle.org.cn/install/quick))。


### 使用例子

使用例子请参考`examples/`目录下对应的[例子目录](examples/)。


## 贡献代码

[PrivBox设计与开发指南](docs/README_cn.md)。

本代码库正在不断开发中，欢迎反馈、报告错误和贡献代码！


## 参考文献

\[ZLH19\] Ligeng Zhu, Zhijian Liu, and Song Han. Deep leakage from gradients. NeurIPS, 2019.

\[HAPC17\] Briland Hitaj, Giuseppe Ateniese, and Fernando P´erez-Cruz. Deep models under the gan: Information leakage from collaborative deep learning. CCS, 2017.

\[YGFJ18\] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. Computer Security Foundations Symposium (CSF), 2018.

\[SZHB19\] Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, Michael Backes. ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models. Network and Distributed Systems Security Symposium (NDSS) 2019.
