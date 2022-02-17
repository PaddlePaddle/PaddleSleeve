# 隐私服务工具使用说明
English | [简体中文](./README_cn.md)

1. Install privbox tool
   
Installation please refer to [PrivBox/README.md-Installation](../../PrivBox/README.md###Installation)

2. Define Dataset

Dataset Definition needs to firstly create a dataset directory，which includes a `.py` file. The file must implement `get_dataset()` function to return dataset object with `paddle.io.Dataset` type. Detail example please see [./example/datasets/cifar10_train/cifar10_train.py](./example/datasets/cifar10_train/cifar10_train.py).

3. Define Model

Similar to Dataset definition, defining a model also needs to create a model directory first, and it contains a `.py` model file and `.pdparams` format parameters file. The model file must implement `get_model()` method to return `paddle.nn.Layer` type model object, which will load `.pdparams`'s params in program. Detail example refers to [./example/models/resnet18_10classes/resnet18.py](./example/models/resnet18_10classes/resnet18.py)。

4. Task Configure
   
We use `.yaml` file for configuring task. The detail configure is shown as follows:
```yaml
- type: MEMBERSHIP_INFERENCE_ATTACK # attack type
  # attack name
  name: BASELINE # see PrivBox/examples/membership_inference/rule_base_with_cifar10/README.md for detail
  # attack args
  args:           
    # test datasets, including [member dataset,  non-member dataset]
    test_datasets: [./example/datasets/cifar10_train,
                    ./example/datasets/cifar10_test]
    # target model
    target_model: ./example/models/resnet18_10classes
    
- type: MEMBERSHIP_INFERENCE_ATTACK # attack type
  # attack name
  name: ML-LEAK # see PrivBox/examples/membership_inference/ml_leaks_with_cifar10_cifar100/README.md for detail
  # attack args
  args:           
    # dataset for target model [member dataset, non-member dataset]
    target_datasets: [./example/datasets/cifar10_train,
                    ./example/datasets/cifar10_test]
    # target model
    target_model: ./example/models/resnet18_10classes
    # shadow model
    shadow_model: ./example/models/resnet34_100classes
    # datasets for training shadow model [member dataset, non-member dataset]
    shadow_datasets: [./example/datasets/cifar100_train,
                    ./example/datasets/cifar100_test]
    # the epoch for training shadow model
    shadow_epoch: 10
    # the learning rate for training shadow model
    shadow_lr: 0.0002
    # the epoch for training attack classifier
    classifier_epoch: 10
    # the learning rate for training attack classifier
    classifier_lr: 0.0002
    # the batch size for training or predition
    batch_size: 128
    # the top k classes of prediction vector used for training attack classifier
    topk: 3

- type: MEMBERSHIP_INFERENCE_ATTACK # attack type
  # attack name
  name: LABEL-ONLY # see PrivBox/examples/membership_inference/label_only_with_cifar10/README_cn.md for detail
  # attack args
  args:           
    # dataset for target model [member dataset, non-member dataset]
    target_datasets: [./example/datasets/cifar10_train,
                      ./example/datasets/cifar10_test]
    # target model
    target_model: ./example/models/resnet18_10classes
    # shadow model
    shadow_model: ./example/models/resnet18_10classes
    # datasets for training shadow model [member dataset, non-member dataset]
    shadow_datasets: [./example/datasets/cifar10_test,
                      ./example/datasets/cifar10_train]
    # the epoch for training shadow model
    shadow_epoch: 10
    # the learning rate for training shadow model
    shadow_lr: 0.0002
    # the epoch for training attack classifier
    classifier_epoch: 10
    # the learning rate for training attack classifier
    classifier_lr: 0.0002
    # the batch size for training or predition
    batch_size: 128
    # data augmentation type, "r" is rotate
    attack_type: r
    # argument for data augmention
    r: 6

```

5. Run Task

```shell
python3 main.py ./example/tasks/mem_inf.yaml
```
