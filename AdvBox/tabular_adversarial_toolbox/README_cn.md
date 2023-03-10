简体中文 | **[English](/AdvBox/tabular_adversarial_toolbox/README.md)**

# Advbox - Tabular Adversarial Toolbox
    表格式对抗工具箱(TAT)是AdvBox的一个子模块，用于为结构化数据训练的模型生成对抗样本。

## 安装
### 要求
- Python >= 3.7
- numpy
- scikit-learn
- xgboost
- pandas

## 开始使用表格对抗工具箱

    这些示例在德国信用数据集上训练XGboost模型，并使用TAT创建对抗示例。在这里，我们训练XGBoost模型提供给TAT预测器，也可以向TAT预测器提供您自己的预训练模型或使用您自己的预测器。参数的选择是为了减少脚本的计算需求，而不是为了精度而优化。

### Dataset
我们的例子是基于**[Statlog (German Credit Data) Data Set](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))**。我们已经将数据下载到**[data/German_Credit_Data](/AdvBox/tabular_adversarial_toolbox/data/German_Credit_Data)**。

German Credit Data的结构如下:

```
German_Credit_Data
|_ german.doc
|_ german.data
|_ german.data-numeric
```

### 示例
所有的例子都在**[examples](/AdvBox/tabular_adversarial_toolbox/examples)**文件夹。

```
cd examples
```

- **[attack_german_credit_data_xgboost.py](/AdvBox/tabular_adversarial_toolbox/examples/attack_german_credit_data_xgboost.py)**演示了一个使用德国信用数据训练XBGoost模型，然后使用TAT来构建预测器，并根据攻击设置生成对抗样本的示例。
  - **命令行参数**
    - `--seed`: 随机种子，默认值:666。
    - `--data_path`: German Credit Data中文件“german.data”的路径。
  - **用法**

    ```
    python attack_german_credit_data_xgboost.py --seed 666 --data_path ../data/German_Credit_Data/german.data
    ```

## 引用

### BibTeX

```bibtex
@misc{Dua:2019,
  author = "Dua, Dheeru and Graff, Casey",
  year = "2017",
  title = "{UCI} Machine Learning Repository",
  url = "http://archive.ics.uci.edu/ml",
  institution = "University of California, Irvine, School of Information and Computer Sciences"
}
```


