# 基于DLG恢复MNIST训练数据例子

如下图所示，本例子模拟在梯度共享的联邦训练场景下，恶意参与者如何通过DLG攻击模块恢复另一方训练数据。

<p align="center">
  <img src="../../../docs/images/dlg_example.png?raw=true" width="700" title="DLG attack in federated learning"/>
</p>

## 运行例子

首先需要按照说明安装privbox（[安装教程](../../../README_cn.md###安装)），安装成功后，运行：
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

**结果示例1**：

输入参数：
```shell
attack_epoch=2000, batch_size=1, learning_rate=0.2, result_dir='./att_results', return_epoch=100, window_size=200
```

输出结果：

```shell
Attack Iteration 0: data_mse_loss = 1.9353874921798706, data_psnr = -2.8676793001011043, data_ssim = 0.009513579308986664, labels_acc = 0.0
Attack Iteration 500: data_mse_loss = 0.0060049062594771385, data_psnr = 22.214937678296963, data_ssim = 0.9713331460952759, labels_acc = 0.0
Attack Iteration 1000: data_mse_loss = 0.00019176446949131787, data_psnr = 37.17231856671547, data_ssim = 0.9990485906600952, labels_acc = 1.0
Attack Iteration 1500: data_mse_loss = 1.5119064300961327e-05, data_psnr = 48.20475085918395, data_ssim = 0.9998847842216492, labels_acc = 1.0
Attack Iteration 1900: data_mse_loss = 2.3815373424440622e-06, data_psnr = 56.231426043727346, data_ssim = 0.9999781847000122, labels_acc = 1.0
```

可以看到，随着攻击迭代轮次增加，攻击恢复的图像与真实图像的MSE、PSNR、SSIM评估指标都达到比较理想的效果，说明攻击比较成功。也可以通过下面两图直观的对比攻击恢复的图像和原图像相似程度（左图为原图，右图为攻击恢复的图）。


<p align="center">
  <img src="../../../docs/images/dlg_target.png?raw=true" width="50" title="DLG target image"/>
  <img src="../../../docs/images/dlg_reconstruct.png?raw=true" width="50" title="DLG attack reconstructed image"/>
</p>