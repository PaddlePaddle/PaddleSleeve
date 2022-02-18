# Usage of Model Privacy Leakage Analysis Tool
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

6. Example Result

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
