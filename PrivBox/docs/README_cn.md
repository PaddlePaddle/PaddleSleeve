# 设计与开发指南


## Attack架构

<p align="center">
  <img src="images/Attacks.png?raw=true" width="700" title="PrivBox Framework">
</p>


## 增加新的Attack

以下代码以增加新的Inference Attack的为例，其他类型Attack类似。

```python

# Step 1. add new attack class which implement abstract attack class
# and implement methods of set_params, infer and evaluate
class NewInferenceAttack(MembershipInferenceAttack):
    def set_params(self, **kwargs):
        super().set_params(kwargs)

    def infer(self, data, **kwargs):
        self._do_infer_attack(data, kwargs)

    def evaluate(self, target, result, metrics, **kwargs):
        self._do_evaluate(target, result, metrics, kwargs)


# Step 2. Add example to show how to conduct the attack in "examples/" dir

```

## 增加新的Metric

```python
# Step 1. add new metric class which implement abstract metric class
# and implement methods of compute
class NewMetric(Metric):
    def compute (self, expected, real):
        self._do_compute_metric(expected, real)

```

## 增加新的Dataset

参考 [paddle增加新的dataset方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/02_data_load_cn.html)。

