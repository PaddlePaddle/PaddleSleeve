# Example of rule based membership inference attack
English | [简体中文](./README_cn.md)

The example is based on a simple rule that an instance is considered as a member if its predict result is correct. This simple method is generally used as the baseline of the membership inference attack. The effect of the attack depends on the level of the overfitting of the model.

## Run example

Install the `privbox` tool ([Installation](../../../README.md###Installation)) firstly, then run the example as:
```shell

python3 baseline_with_cifar10.py

```

The example provides the following parameters that the user can customize the settings.

- `--batch_size` (int, default=128): The batch size for training target model.
- `--train_epoch` (int, default=10): The epoch for training target model
- `--train_lr` (float, default=0.0002): The learning rate for training target model。


**Example Result 1**：

Input Parameters：

```shell
batch_size=128, train_epoch=10, train_lr=0.0002
```

Evaluation Result：
```shell
Evaluate result of baseline membership attack on cifar10 is: acc: 0.72394,
          auc: 0.7993062947773256, precision: 0.97484， recall: 0.6491143960580636
```