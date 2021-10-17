# 通过压缩梯度来保护隐私

梯度压缩算法把梯度小于阈值的梯度去除掉，只保留少量重要的梯度，这样的算法可以应用到联邦学习中来减少梯度的传输通信量，同时对于基于梯度的攻击也起到一定的防护作用。
目前，比较广泛使用的梯度压缩算法是DGC（Deep Gradient Compress）算法，其算法流程如下图所示[1]。

<p align="center">
  <img src="../docs/images/dgc.png?raw=true" width="700" title="DGC algorithm"/>
</p>

我们将其封装成paddle优化器，使用方法跟paddle的优化器一样，具体例子如`dgc_demo.py`所示。

同时，`dgc_demo.py`还加入了DLG攻击选项（攻击详见文件`PrivBox/inversion/dlg.py`），方便从攻击的角度来衡量梯度压缩的隐私保护效果。例子具体使用方法如下：

## 运行例子

```shell

python3 dgc_demo.py

```

例子提供以下参数，用户可以自定义设置

- `--train_epoch` (int, default=2): 正常训练的迭代轮数。
- `--train_batch_size` (int, default=64): 正常模型训练的batch size。
- `--train_lr` (float, default=0.001): 模型训练的学习率。
- `--sparsity` (float, default=0.8): 梯度稀疏度，设置压缩后梯度的稀疏度。
- `--use_dgc` (bool, default=True): 是否使用DGCMomentum优化器，为False时使用Momentum优化器。
- `--dlg_attack` (bool, default=False): 是否执行DLG攻击。


启动DLG攻击时，还可以通过以下参数控制DLG攻击：

- `--attack_batch_size` (int, default=1): 攻击训练数据的batch size, 也是要恢复数据量 （注意目前batch size大于1时，恢复的图像会有重叠）。
- `--attack_epoch` (int, default=2000): DLG攻击的epoch数量。
- `--attack_lr` (float, default=0.2): DLG攻击过程的学习率。
- `--result_dir` (str, default='./att_results'): 攻击结果保存目录。
- `--return_epoch` (int, default=100): 每多少个attack_epcoh保存一个结果。
- `--window_size` (int, default=200): DLG论文提出，当batch_size大于1时，交替地更新batch中的样本会更快地收敛，window_size即为每多少个样本更新后再更新另一个样本。


默认地，程序运行结束后，结果保存在运行目录所在的`att_results/`目录下, 用户也可以通过`--result_dir`自行指定保存位置。

**结果示例1**：

输入参数：
```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=True, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.001, use_dgc=True, window_size=200
```

输出结果：

```shell
Attack Iteration 0: data_mse_loss = 1.8928502798080444, data_psnr = -2.77116263607378, data_ssim = 0.01727583445608616, labels_acc = 0.0
Attack Iteration 500: data_mse_loss = 0.7674962282180786, data_psnr = 1.149237501414605, data_ssim = 0.10738208144903183, labels_acc = 0.0
Attack Iteration 1000: data_mse_loss = 0.148170605301857, data_psnr = 8.292379449491232, data_ssim = 0.42095354199409485, labels_acc = 0.0
Attack Iteration 1500: data_mse_loss = 0.03237803652882576, data_psnr = 14.897494913007217, data_ssim = 0.6388757824897766, labels_acc = 0.0
Attack Iteration 1900: data_mse_loss = 0.024529650807380676, data_psnr = 16.103086341624326, data_ssim = 0.7362926006317139, labels_acc = 0.0
```

**结果示例2**：

输入参数：
```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=True, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.01, use_dgc=False, window_size=200
```

输出结果：

```shell
Attack Iteration 0: data_mse_loss = 1.8898268938064575, data_psnr = -2.7642202506864306, data_ssim = 0.010832361876964569, labels_acc = 0.0
Attack Iteration 500: data_mse_loss = 0.08307313174009323, data_psnr = 10.805394169357015, data_ssim = 0.543327808380127, labels_acc = 1.0
Attack Iteration 1000: data_mse_loss = 0.003942424431443214, data_psnr = 24.04236622494114, data_ssim = 0.9001388549804688, labels_acc = 1.0
Attack Iteration 1500: data_mse_loss = 0.0005089049809612334, data_psnr = 32.933632984028065, data_ssim = 0.983788251876831, labels_acc = 1.0
Attack Iteration 1900: data_mse_loss = 0.00013265199959278107, data_psnr = 38.772861990886206, data_ssim = 0.9935102462768555, labels_acc = 1.0
```

从例1和例2可以看到，随着攻击迭代轮次增加，启用DGCMomentum优化器时，DLG攻击评估指标MSE、PSNR、SSIM都比没启用时差，即表示对DLG攻击有防护效果。


**结果示例3**：

输入参数：
```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=False, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.01, use_dgc=False, window_size=200
```

输出结果：

```shell
epoch 0, batch id 0, training loss 3.135075569152832, acc 0.015625.
epoch 0, batch id 500, training loss 0.24693243205547333, acc 0.8592190618762475.
epoch 0, batch id 900, training loss 0.29456984996795654, acc 0.8744797447280799.
epoch 1, batch id 0, training loss 0.37163984775543213, acc 0.875.
epoch 1, batch id 500, training loss 0.6617305874824524, acc 0.906312375249501.
epoch 1, batch id 900, training loss 0.2521766424179077, acc 0.9057991120976693.
```

**结果示例4**：

输入参数：
```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=False, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.001, use_dgc=True, window_size=200
```

输出结果：

```shell
epoch 0, batch id 0, training loss 3.250225782394409, acc 0.078125.
epoch 0, batch id 500, training loss 12.515483856201172, acc 0.4638223552894212.
epoch 0, batch id 900, training loss 4.712625503540039, acc 0.6165024972253053.
epoch 1, batch id 0, training loss 4.502931118011475, acc 0.765625.
epoch 1, batch id 500, training loss 2.019190788269043, acc 0.8789608283433133.
epoch 1, batch id 900, training loss 3.7424607276916504, acc 0.8587506936736959.
```

从例3和例4可以看到，DGCMomentum训练时，最大acc可以到达0.879, 而Momentum可以到达0.906，有一定的精度下降。

**参考文献：**

[1] Lin Y, Han S, Mao H, et al. Deep gradient compression: Reducing the communication bandwidth for distributed training[J]. arXiv preprint arXiv:1712.01887, 2017.