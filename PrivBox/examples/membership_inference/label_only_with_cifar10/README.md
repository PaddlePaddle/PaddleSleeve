# Example of label-only membership inference attack
English | [简体中文](./README_cn.md)

The example illustrates how to use the label-only membership inference attack module to attack a target model. The attack process of the example is shown in the figure below.

<p align="center">
  <img src="../../../docs/images/labelonly_example.png?raw=true" width="500" title="Label-only membershi attack example"/>
</p>

## Run example

```shell
d PaddleSleeve/PrivBox/examples/membership_inference/label_only_with_cifar10
python3 label_only_with_cifar10.py

```

The example provides the following parameters that the user can customize the settings.

- `--batch_size` (int, default=128): The batch size for training model.
- `--target_epoch` (int, default=10): The epoch for training target model.
- `--shadow_epoch` (int, default=10): The epoch for training shadow model.
- `--classifier_epoch` (int, default=10): The epoch for training  classifier.
- `--target_lr` (float, default=0.0002): The learning rate for training target model.
- `--shadow_lr` (float, default=0.0002): The learning rate for training shadow model.
- `--classifier_lr` (float, default=0.0002): The learning rate for training classifier.
- `--shadow_dataset` (str, default=cifar10): Shadow dataset(cifar10).
- `--target_dataset` (str, default=cifar10): Target dataset(cifar10).
- `--shadow_model` (str, default=resnet18): Shadow model(resnet18 or resnet34).
- `--attack_type` (str, default=r): Type of attack to perform, r is rotation.
- `--r` (int, default=6): param in rotation attack if used.


**Example Result 1**：

Input Parameters：

```shell
attack_type='r', batch_size=128, classifier_epoch=10, classifier_lr=0.0002, r=6, shadow_dataset='cifar10', shadow_epoch=10, shadow_lr=0.0002, shadow_model='resnet18', target_dataset='cifar10', target_epoch=10, target_lr=0.0002
```

Command Line
```shell
python label_only_with_cifar10.py --attack_type='r' --batch_size=128 --classifier_epoch=10 --classifier_lr=0.0002 --r=6 --shadow_dataset='cifar10' --shadow_epoch=10 --shadow_lr=0.0002 --shadow_model='resnet18' --target_dataset='cifar10' --target_epoch=10 --target_lr=0.0002
```

Evaluation Result：
```shell
Evaluate result of Label-only membership attack is: acc: 0.8794666666666666,
          auc: 0.74830688, precision: 0.8996971849414931， recall: 0.96264
```
