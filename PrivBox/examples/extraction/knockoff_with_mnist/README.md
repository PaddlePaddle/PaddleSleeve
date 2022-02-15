# Example for Knockoff model extraction on model of MNIST
English | [简体中文](./README_cn.md)

As shown below, this example simulates how to use Knockoff Model Extraction modulus to extract a victim's model.

1. Firstly, A model $F_V$ is trained by victim using his dataset. Then deploy the model. Others can access the model though its predict APIs.
2. Adversary chooses the dataset and model architecture, and uses Knockoff Model Extraction modulus to extract victim model. Output model is $F_A$
3. Optionally, adversary can use test dataset to evalute the accuracy for model $F_A$ and model $F_A$ (optionally).

<p align="center">
  <img src="../../../docs/images/knockoff_example.png?raw=true" width="700" title="Knockoff Modle extraction Attack Framework"/>
</p>

## Run example

Install the `privbox` tool ([Installation](../../../README.md###Installation)) firstly, then run the example as:

```shell

python3 knockoff_extraction_with_mnist.py

```

After program finished, the accuracy for victim's model and adversary's model is print as follows:

```
Victim model's evaluate result:  [xxx]
Knockoff model's evaluate result:  [xxx]
```

The example provides the following parameters that the user can customize the settings.

- `--batch_size` (int, default=128): The batch size of training and predict.
- `--epochs` (int, default=2): The iterations of training for victim and adversary.
- `--learning_rate` (float, default=0.01): The learning rate of training for victim and adversary.
- `--num_queries` (int, default=2000): The number of queries allowed for adversary.
- `--knockoff_net` (str, default="linear"): The network for knockoff model, can be chosen from 'linear' and 'resnet'.
- `--knockoff_dataset` (str, default="mnist"): The dataset for training knockoff model, can be chosen from 'mnist' (100% labels overlap) or 'fmnist' (0% labels overlap).
- `--policy` (str, default="random"):Sampling policy. One can choose "random" or "adaptive" policy. "random" sampling policy randomly samples input data to query victim model, while "adaptive" sampling policy samples input data based on its feedback of rewards.
- `--reward` (str, default="all"): Reward strategy, only for "adaptive" policy. One can choose "certainty", "diversity", "loss" and "all". "certainty" reward is used margin-based certainty measure, "diversity" is used for preventing the degenerate case of image exploitation, "loss" strategy reward high loss images, and "all" strategy uses all three reward strategies.

**Example Result 1**：

Input Parameters:
```shell
batch_size=128, epochs=2, knockoff_net='linear', knockoff_dataset='fmnist', learning_rate=0.01, num_queries=40000, policy='random', reward='all'
```

Evaluation Result
```
Victim model's evaluate result:  [0.906]
Knockoff model's evaluate result:  [0.862]
```

**Example Result 2**：

Input Parameters：
```shell
batch_size=128, epochs=2, knockoff_dataset='mnist', knockoff_net='linear', learning_rate=0.01, num_queries=6000, policy='adaptive', reward='all'
```

Evaluation Result：
```
Victim model's evaluate result:  [0.8905]
Knockoff model's evaluate result:  [0.886]
```