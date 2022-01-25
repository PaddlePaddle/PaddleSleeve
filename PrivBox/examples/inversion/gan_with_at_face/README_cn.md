# 基于GAN恢复AT&T人脸训练数据例子

如下图所示，本例子模拟在参数共享的联邦训练场景下，恶意参与者如何通过GAN攻击模块恢复另一方训练数据。

<p align="center">
  <img src="../../../docs/images/gan_example.png?raw=true" width="700" title="GAN attack in federated learning">
</p>


## 运行例子

首先需要按照说明安装privbox（[安装教程](../../../README_cn.md###安装)），安装成功后，运行：
```shell

python3 gan_inversion_with_at_face.py

```

例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=32): 训练数据的batch size。
- `--attack_epoch` (int, default=100): GAN攻击的训练epoch数量。
- `--target_label` (int, default=1): 攻击目标标签，要恢复的数据标签。
- `--learning_rate_real` (float, default=0.0002): 实际联邦训练的学习率。
- `--learning_rate_fake` (float, default=0.0002): 攻击者用虚假数据训练真实模型的学习率。
- `--learning_rate_gen` (float, default=0.0002): 攻击者训练GAN生成器的学习率。
- `--result_dir` (str, default='./att_results'): 攻击结果保存目录。
- `--num_pic_save` (int, default=4): 每个epoch存储多少个攻击结果图片。


默认地，程序运行结束后，结果保存在运行目录所在的`att_results/`目录下, 用户也可以通过`--result_dir`自行指定保存位置。

**结果示例1**：

输入参数：
```shell
attack_epoch=100, batch_size=32, learning_rate_fake=0.0002, learning_rate_gen=0.0002, learning_rate_real=0.0002, num_pic_save=4, result_dir='./att_results', target_label=1
```

攻击结果：

目标图像与攻击恢复的图像如下图所示（左右为目标图像，右图为攻击恢复的图像）。

<p align="center">
  <img src="../../../docs/images/gan_target.png?raw=true" width="50" title="GAN target image"/>           
  <img src="../../../docs/images/gan_reconstruct.png?raw=true" width="50" height="60" title="GAN attack reconstructed image"/>
</p>