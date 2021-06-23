# 基于DLG恢复MNIST训练数据例子

如下图所示，本例子模拟在梯度共享的联邦训练场景下，恶意参与者如何通过DLG攻击模块恢复另一方训练数据。

<p align="center">
  <img src="../../../docs/images/dlg_example.png?raw=true" width="700" title="DLG attack in federated learning"/>
</p>

## 运行例子

```shell

python3 dlg_inversion_with_mnist.py

```

例子提供以下参数，用户可以自定义设置

- `--batch_size` (int, default=1): 训练数据的batch size, 也是要恢复数据量 （注意目前batch size大于1时，恢复的图像会有重叠）。
- `--attack_epoch` (int, default=2000): DLG攻击的epoch数量。
- `--learning_rate` (float, default=0.2): DLG攻击过程的学习率。
- `--result_dir` (str, default='./att_results'): 攻击结果保存目录。
- `--return_epoch` (int, default=100): 每多少个attack_epcoh保存一个结果。
- `--window_size` (int, default=200): DLG论文提出，当batch_size大于1时，交替地更新batch中的样本会更快地收敛，window_size即为每多少个样本更新后再更新另一个样本。


默认地，程序运行结束后，结果保存在运行目录所在的`att_results/`目录下, 用户也可以通过`--result_dir`自行指定保存位置。