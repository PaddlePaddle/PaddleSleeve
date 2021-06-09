# Development Guide
English | [简体中文](./README_cn.md)

## Structure of Codebase
```
PrivBox
├── dataset                     \\ Directory for datasets
│
├── docs                        \\ Directory for document
│
├── examples                    \\ Examples directory
│
├── extraction                  \\ Model extraction attack modulus
│
├── inference
│   ├── membership_inference    \\ Membership Inference attack modulus
│   └── property_inference      \\ Property Inference attack modulus
│
├── inversion                   \\ Model inversion attack modulus
│
├── metrics                     \\ Metric modulus, for attack evaluation
│
└── tests
    └── unittests               \\ Unittests

```
## Attack Classes

<p align="center">
  <img src="images/Attacks.png?raw=true" width="700" title="PrivBox Framework">
</p>


## Adding a New Attack

Following codes show how to add a new inference attack. Other attacks are similar.

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

## Adding a New Metric

A metric (such as Accuracy, AUC) is used for evaluating the effect of an attack. Following codes show how to add a new metric.

```python
# Step 1. add new metric class which implement abstract metric class
# and implement methods of compute
class NewMetric(Metric):
    def compute (self, expected, real):
        self._do_compute_metric(expected, real)

```

## Adding a New Dataset

Same as  [paddle add new dataset](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/io/Dataset_en.html)

