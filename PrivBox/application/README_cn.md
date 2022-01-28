# 隐私服务工具使用说明

1. 安装privbox工具
   
安装教程请参考[PrivBox/README_cn.md-安装](../../PrivBox/README_cn.md###安装)

2. 定义数据集
   
数据集定义参考需要新建一个数据集目录，目录中包含一个`.py`文件，文件内需要定义一个`get_dataset()`函数来返回paddle.io.Dataset类型的数据集对象。具体可以参考例子[./example/datasets/cifar10_train/cifar10_train.py](./example/datasets/cifar10_train/cifar10_train.py)。

3. 定义模型
   
模型定义通定义数据集类似。需要首先新建一个模型目录，目录包含一个`.py`模型定义文件和`.pdparams`模型参数文件，模型定义文件需要实现`get_model()`方法来返回paddle.nn.Layer类型的模型对象。该模型对象的参数后续会自动加载`.pdparams`模型参数。具体可以参考例子[./example/models/resnet18_10classes/resnet18.py](./example/models/resnet18_10classes/resnet18.py)。

4. 配置任务
   
任务配置使用`.yaml`格式文件，目前主要包含两种攻击，攻击配置及说明如下：
```yaml
- type: MEMBERSHIP_INFERENCE_ATTACK # 攻击类型
  # 攻击的名称
  name: BASELINE # 参考 PrivBox/examples/membership_inference/rule_base_with_cifar10/README_cn.md 的攻击说明
  # 攻击相关参数
  args:           
    # 测试数据集，包含目标模型的[成员数据集， 非成员数据集]
    test_datasets: [./example/datasets/cifar10_train,
                    ./example/datasets/cifar10_test]
    # 目标模型
    target_model: ./example/models/resnet18_10classes
    
- type: MEMBERSHIP_INFERENCE_ATTACK # 攻击类型
  # 攻击的名称
  name: ML-LEAK # 参考 PrivBox/examples/membership_inference/ml_leaks_with_cifar10_cifar100/README_cn.md 的攻击说明
  # 攻击相关参数
  args:           
    # 目标模型的[成员数据集， 非成员数据集]
    target_datasets: [./example/datasets/cifar10_train,
                    ./example/datasets/cifar10_test]
    # 目标模型
    target_model: ./example/models/resnet18_10classes
    # 使用的影子模型名称
    shadow_model: ./example/models/resnet34_100classes
    # 用来训练影子模型的[成员数据集、非成员数据集]
    shadow_datasets: [./example/datasets/cifar100_train,
                    ./example/datasets/cifar100_test]
    # 影子模型的训练epoch数
    shadow_epoch: 10
    # 影子模型训练用的学习率
    shadow_lr: 0.0002
    # 攻击模型训练的epoch
    classifier_epoch: 10
    # 攻击模型使用的lr
    classifier_lr: 0.0002
    # 训练、预测的batch size
    batch_size: 128
    # 训练攻击模型使用的输出向量top k个
    topk: 3

```