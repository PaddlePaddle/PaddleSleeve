# 隐私服务工具使用说明

1. 安装privbox工具
   
安装教程请参考[PrivBox/README_cn.md-安装](../../PrivBox/README_cn.md###安装)

2. 定义数据集
   
数据集定义参考需要新建一个数据集目录，目录中包含一个`.py`文件，文件内需要定义一个`get_dataset()`函数来返回paddle.io.Dataset类型的数据集对象。具体可以参考例子[./example/datasets/cifar10_train/cifar10_train.py](./example/datasets/cifar10_train/cifar10_train.py)。

3. 定义模型
   
模型定义同定义数据集类似。需要首先新建一个模型目录，目录包含一个`.py`模型定义文件和`.pdparams`模型参数文件，模型定义文件需要实现`get_model()`方法来返回paddle.nn.Layer类型的模型对象。该模型对象的参数后续会自动加载`.pdparams`模型参数。具体可以参考例子[./example/models/resnet18_10classes/resnet18.py](./example/models/resnet18_10classes/resnet18.py)。

4. 配置任务
   
任务配置使用`.yaml`格式文件，目前主要包含三种攻击，攻击配置及说明如下：
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

- type: MEMBERSHIP_INFERENCE_ATTACK # 攻击类型
  # 攻击的名称
  name: LABEL-ONLY # 参考 PrivBox/examples/membership_inference/label_only_with_cifar10/README_cn.md 的攻击说明
  # 攻击相关参数
  args:           
    # 目标模型的[成员数据集， 非成员数据集]
    target_datasets: [./example/datasets/cifar10_train,
                      ./example/datasets/cifar10_test]
    # 目标模型
    target_model: ./example/models/resnet18_10classes
    # 使用的影子模型名称
    shadow_model: ./example/models/resnet18_10classes
    # 用来训练影子模型的[成员数据集、非成员数据集]
    shadow_datasets: [./example/datasets/cifar10_test,
                      ./example/datasets/cifar10_train]
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
    # 训练攻击模型使用的数据增强方案
    attack_type: r
    # 数据增强参数
    r: 6
```

5. 运行任务

```shell
python3 main.py ./example/tasks/mem_inf.yaml
```

6. 示例结果

```
Model Privacy Leakage Analysis Report
Models:
	- name: resnet18, train_acc: 0.82344, test_acc: 0.5927
Datasets:
	- name: cifar10_train, is_member_dataset: true, length: 50000
	- name: cifar10_test, is_member_dataset: false, length: 10000
Attacks:
	- name: Baseline Attack
	  attack description: An instance is considered as a member if its predict result is correct. It only requires data with labels.
	  attack results: acc = 0.7540833333333333, auc = 0.5949353236670549, precision = 0.82344, recall = 0.8741586870209559
	- name: ML-LEAK Attack
	  attack description: A membership inference attack based on auxiliary dataset, shadow model and prediction confidence
	  attack results: acc = 0.55245, auc = 0.562069685, precision = 0.8583283599056469, recall = 0.60404
	- name: LABEL-ONLY Attack
	  attack description: A membership inference attack based on auxiliary dataset, shadow model and prediction label
	  attack results: acc = 0.6994, auc = 0.631975736, precision = 0.8763995324068061, recall = 0.67474
Summary:
	WARNING! Your model has risk of membership inference attacks, they are: 
	1, Baseline Attack (MIDDLE risk)
	2, ML-LEAK Attack (MIDDLE risk)
	3, LABEL-ONLY Attack (MIDDLE risk)

	There are some defense recommends you can implement to prevent membership information leakage:
	1, Differential Privacy. A model is trained in a differentially private manner, the learned model does not learn or remember any specific user’s details. Thus, differential privacy naturally counteracts membership inference.
	2, Confidence Masking. Confidence score masking method aims to hide the true confidence score returned by the target model and thus mitigates the effectiveness of membership inference attack. The defense methods belonging to this stratety includes restricting the prediction vector to top k classes, rounding prediction vector to small decimals, only returning labels, or adding crafted noise to prediciton vector.
	3, Knowledge Distillation. Knowledge distillation uses the outputs of a large teacher model to train a smaller one, in order to transfer knowledge from the large model to the small one. This strategy is to restrict the protected classifier’s direct access to the private training dataset, thus significantly reduces the membership information leakage.
	4, Regularization. Overfitting is the main factor that contributes to membership inference attack. Therefore, regularization techniques that can reduce the overfitting of ML models can be leveraged to defend against the attack. Regularization techniques including L2-norm regularization, dropout, data argumentation, model stacking, early stopping, label smoothing and adversarial regularization can be used as defense methods for defenses.
```