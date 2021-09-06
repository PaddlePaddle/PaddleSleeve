# Privacy Preservation by Gradient Compress
English | [简体中文](./README_cn.md)

Gradient compress algorithm only uses important gradients (larger than the threshold) to update network parameters. Grandient compress not only can reduce commmunication cost in federated learning, but also can prevent gradient-based attack (such as DLG attack, see `PrivBox/inversion/dlg.py` file).

DGC (Deep Gradient Compress) is one of the famous gradient compress algorithm. Detailed algorithm is shown as follows [1].

<p align="center">
  <img src="../docs/images/dgc.png?raw=true" width="700" title="DGC algorithm"/>
</p>

We implement DGC as a Paddle optimizer (DGCMomentum) just as other Paddle optimizer like SGD. We also give an example for DGCMomentum optimizer. Meanwhile, DLG attack is append for evaluating the defense of gradient compress.

## Run example

```shell

python3 dgc_demo.py

```

The example provides the following parameters that the user can customize the settings.

- `--train_epoch` (int, default=2): The iteration for normal model training.
- `--train_batch_size` (int, default=64): The batch size of normal model training.
- `--train_lr` (float, default=0.001): The learning rate for model training.
- `--sparsity` (float, default=0.8): The gradient sparsity, the degree of gradient compress.
- `--use_dgc` (bool, default=True): Whether to use `DGCMomentum` optimizer, set `False` to use `Momentum` optimizer.
- `--dlg_attack` (bool, default=False): whether to launch DLG attack.

Following parameters can be set when `--dlg_attack` is set to `True`:

- `--attack_batch_size` (int, default=1): batch size of attack training data.
- `--attack_epoch` (int, default=2000): iterations of DLG attack.
- `--attack_lr` (float, default=0.2): leaning rate of DLG.
- `--result_dir` (str, default='./att_results'): results saving dir.
- `--return_epoch` (int, default=100): save a result per `return_epoch` epoch.
- `--window_size` (int, default=200): The DLG paper proposes that when `attack_batch_size` is greater than 1, updating the samples in batch alternately converges faster, and window_size decides the number of epoch to update other samples.

By default, the results are saved in the 'att_results/' directory where the running directory is located. User can also specify the location by `--result_dir`.


**Example Result 1**:

Input Parameters:

```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=True, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.001, use_dgc=True, window_size=200
```

Output Result:

```shell
Attack Iteration 0: data_mse_loss = 1.8928502798080444, data_psnr = -2.77116263607378, data_ssim = 0.01727583445608616, labels_acc = 0.0
Attack Iteration 500: data_mse_loss = 0.7674962282180786, data_psnr = 1.149237501414605, data_ssim = 0.10738208144903183, labels_acc = 0.0
Attack Iteration 1000: data_mse_loss = 0.148170605301857, data_psnr = 8.292379449491232, data_ssim = 0.42095354199409485, labels_acc = 0.0
Attack Iteration 1500: data_mse_loss = 0.03237803652882576, data_psnr = 14.897494913007217, data_ssim = 0.6388757824897766, labels_acc = 0.0
Attack Iteration 1900: data_mse_loss = 0.024529650807380676, data_psnr = 16.103086341624326, data_ssim = 0.7362926006317139, labels_acc = 0.0
```

**Example Result 2**:

Input Parameters:

```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=True, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.01, use_dgc=False, window_size=200
```

Output Result:

```shell
Attack Iteration 0: data_mse_loss = 1.8898268938064575, data_psnr = -2.7642202506864306, data_ssim = 0.010832361876964569, labels_acc = 0.0
Attack Iteration 500: data_mse_loss = 0.08307313174009323, data_psnr = 10.805394169357015, data_ssim = 0.543327808380127, labels_acc = 1.0
Attack Iteration 1000: data_mse_loss = 0.003942424431443214, data_psnr = 24.04236622494114, data_ssim = 0.9001388549804688, labels_acc = 1.0
Attack Iteration 1500: data_mse_loss = 0.0005089049809612334, data_psnr = 32.933632984028065, data_ssim = 0.983788251876831, labels_acc = 1.0
Attack Iteration 1900: data_mse_loss = 0.00013265199959278107, data_psnr = 38.772861990886206, data_ssim = 0.9935102462768555, labels_acc = 1.0
```

From Example 1 and 2, it can be seen that the attack metrics of `MSE`, `PSNR` and `SSIM` are worse significantly when using `DGCMomentum` optimizer, which means gradient compress strategy can defense DLG attack.

**Example Result 3**：

Input Parameters:

```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=False, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.01, use_dgc=False, window_size=200
```

Output Result:

```shell
epoch 0, batch id 0, training loss 3.135075569152832, acc 0.015625.
epoch 0, batch id 500, training loss 0.24693243205547333, acc 0.8592190618762475.
epoch 0, batch id 900, training loss 0.29456984996795654, acc 0.8744797447280799.
epoch 1, batch id 0, training loss 0.37163984775543213, acc 0.875.
epoch 1, batch id 500, training loss 0.6617305874824524, acc 0.906312375249501.
epoch 1, batch id 900, training loss 0.2521766424179077, acc 0.9057991120976693.
```

**Example Result 4**：

Input Parameters:

```shell
attack_batch_size=1, attack_epoch=2000, attack_lr=0.2, dlg_attack=False, result_dir='./att_results', return_epoch=100, sparsity=0.8, train_batch_size=64, train_epoch=2, train_lr=0.001, use_dgc=True, window_size=200
```

Output Result:

```shell
epoch 0, batch id 0, training loss 3.250225782394409, acc 0.078125.
epoch 0, batch id 500, training loss 12.515483856201172, acc 0.4638223552894212.
epoch 0, batch id 900, training loss 4.712625503540039, acc 0.6165024972253053.
epoch 1, batch id 0, training loss 4.502931118011475, acc 0.765625.
epoch 1, batch id 500, training loss 2.019190788269043, acc 0.8789608283433133.
epoch 1, batch id 900, training loss 3.7424607276916504, acc 0.8587506936736959.
```

Example 3 and 4 show that DGCMomentum would slightly decrease accuracy of model training (DGCMomentum: 0.879 vs Momentum: 0.906).

**Reference:**

[1] Lin Y, Han S, Mao H, et al. Deep gradient compression: Reducing the communication bandwidth for distributed training[J]. arXiv preprint arXiv:1712.01887, 2017.
