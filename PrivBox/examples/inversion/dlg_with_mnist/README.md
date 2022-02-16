# Example for using DLG to reconstruct MNIST data
English | [简体中文](./README_cn.md)

As shown below, this example simulates how malicious participants recover training data from the other party through the DLG attack module in a federated training scenario that sharing model gradient.

<p align="center">
  <img src="../../../docs/images/dlg_example.png?raw=true" width="700" title="DLG attack in federated learning">
</p>

## Run example

Install the `privbox` tool ([Installation](../../../README.md###Installation)) firstly, then run the example as:
```shell

python3 dlg_inversion_with_mnist.py

```

The example provides the following parameters that the user can customize the settings.

- `--batch_size` (int, default=1): batch size of training data.
- `--attack_epoch` (int, default=2000): iterations of DLG attack.
- `--learning_rate` (float, default=0.2): leaning rate of DLG.
- `--result_dir` (str, default='./att_results'): results saving dir.
- `--return_epoch` (int, default=100): save a result per `return_epoch` epoch.
- `--window_size` (int, default=200): The DLG paper proposes that when batch_size is greater than 1, updating the samples in batch alternately converges faster, and window_size decides the number of epoch to update other samples.


By default, the results are saved in the 'att_results/' directory where the running directory is located. User can also specify the location by `--result_dir`.


**Example Result 1**：

Input Parameters：

```shell
attack_epoch=2000, batch_size=1, learning_rate=0.2, result_dir='./att_results', return_epoch=100, window_size=200
```

Evaluation Result：

```shell
Attack Iteration 0: data_mse_loss = 1.9353874921798706, data_psnr = -2.8676793001011043, data_ssim = 0.009513579308986664, labels_acc = 0.0
Attack Iteration 500: data_mse_loss = 0.0060049062594771385, data_psnr = 22.214937678296963, data_ssim = 0.9713331460952759, labels_acc = 0.0
Attack Iteration 1000: data_mse_loss = 0.00019176446949131787, data_psnr = 37.17231856671547, data_ssim = 0.9990485906600952, labels_acc = 1.0
Attack Iteration 1500: data_mse_loss = 1.5119064300961327e-05, data_psnr = 48.20475085918395, data_ssim = 0.9998847842216492, labels_acc = 1.0
Attack Iteration 1900: data_mse_loss = 2.3815373424440622e-06, data_psnr = 56.231426043727346, data_ssim = 0.9999781847000122, labels_acc = 1.0
```

It can be seen from the output that the MSE、PSNR、SSIM loss are really perfect for target image and reconstructed image. We can also intuitively find their similarities from following two images (left is target image, right is reconstructed image). 


<p align="center">
  <img src="../../../docs/images/dlg_target.png?raw=true" width="50" title="DLG target image"/>           
  <img src="../../../docs/images/dlg_reconstruct.png?raw=true" width="50" title="DLG attack reconstructed image"/>
</p>