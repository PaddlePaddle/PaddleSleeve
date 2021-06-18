# Example of rule based membership inference attack
English | [简体中文](./README_cn.md)

The example illustrates how to use the ML-Leaks membership inference attack module to attack a target model. The attack process of the example is shown in the figure below.

<p align="center">
  <img src="../../../docs/images/ml_leak_example.png?raw=true" width="500" title="ML-Leaks membershi attack example"/>
</p>

## Run example

```shell

python3 ml_leaks_with_cifar10_cifar100.py

```

The example provides the following parameters that the user can customize the settings.

- `--batch_size` (int, default=128): The batch size for training model.
- `--target_epoch` (int, default=10): The epoch for training target model.
- `--shadow_epoch` (int, default=10): The epoch for training shadow model.
- `--classifier_epoch` (int, default=10): The epoch for training ML-Leaks classifier.
- `--target_lr` (float, default=0.0002): The learning rate for training target model.
- `--shadow_lr` (float, default=0.0002): The learning rate for training shadow model.
- `--classifier_lr` (float, default=0.0002): The learning rate for training ML-Leaks classifier.
- `--topk` (int, default=10): The top k predict results for training ML-Leaks classifier.
- `--shadow_dataset` (str, default=cifar10): Shadow dataset(cifar10 or cifar100).
- `--target_dataset` (str, default=cifar10): Target dataset(cifar10 or cifar100).
- `--shadow_model` (str, default=resnet18): Shadow model(resnet18 or resnet34).