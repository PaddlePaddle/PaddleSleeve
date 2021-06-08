# Example for using DLG to reconstruct MNIST data
English | [简体中文](./README_cn.md)

This example simulates how malicious participants recover training data from the other party through the DLG attack module in a  federated training scenario that sharing model gradient.

## Run example

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