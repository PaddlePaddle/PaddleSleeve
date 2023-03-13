简体中文 | **[English](/AdvBox/tabular_adversarial_toolbox/README.md)**

# Advbox - Tabular Data
AdvBox -tabular_data是AdvBox的子模块，用于为结构化数据机器学习模型生成对抗样本。

## 安装
### 要求
- Python >= 3.7
- numpy
- scikit-learn
- xgboost
- pandas

## 开始使用Advbox-tabular_data

这些示例在German Credit Data上训练XGboost模型，并使用Advbox-tabular_data创建对抗示例。在这里，我们训练XGBoost模型提供给Advbox-tabular_data预测器，也可以向Advbox-tabular_data的预测器提供您自己的预训练模型或使用您自己的预测器。参数的选择是为了减少脚本的计算需求，而不是为了精度而优化。

### Dataset
我们的例子是基于 **[Statlog (German Credit Data) Data Set](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))**。我们已经将数据下载到 **[data/German_Credit_Data](/AdvBox/tabular_adversarial_toolbox/data/German_Credit_Data)**。

German Credit Data的结构如下:

```
German_Credit_Data
|_ german.doc
|_ german.data
|_ german.data-numeric
```

### 示例
所有的例子都在 **[examples](/AdvBox/tabular_adversarial_toolbox/examples)** 文件夹。

```
cd examples
```

- **[attack_german_credit_data_xgboost.py](/AdvBox/tabular_adversarial_toolbox/examples/attack_german_credit_data_xgboost.py)** 演示了一个使用German Credit Data训练XBGoost模型，然后使用Advbox-tabular_data来构建预测器，并根据攻击设置生成对抗样本的示例。
  - **命令行参数**
    - `--seed`: 随机种子，默认值:666。
    - `--data_path`: German Credit Data中文件“german.data”的路径。
  - **用法**

    ```
    python attack_german_credit_data_xgboost.py --seed 666 --data_path ../data/German_Credit_Data/german.data
    ```
  - **结果**

    | Samples | Status&nbsp;of&nbsp;existing&nbsp;checking&nbsp;account | Duration&nbsp;in&nbsp;month | Credit&nbsp;history | Purpose | Credit&nbsp;amount | Savings&nbsp;account/bonds | Present&nbsp;employment&nbsp;since | Installment&nbsp;rate&nbsp;in&nbsp;percentage&nbsp;of&nbsp;disposable&nbsp;income | Personal&nbsp;status&nbsp;and&nbsp;sex | Other&nbsp;debtors/guarantors | Present&nbsp;residence&nbsp;since | Property | Age&nbsp;in&nbsp;years | Other&nbsp;installment&nbsp;plans | Housing | Number&nbsp;of&nbsp;existing&nbsp;credits&nbsp;at&nbsp;this&nbsp;bank | Job | Number&nbsp;of&nbsp;people&nbsp;being&nbsp;liable&nbsp;to&nbsp;provide&nbsp;maintenance&nbsp;for | Telephone | foreign&nbsp;worker |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | Original&nbsp;sample | <&nbsp;0&nbsp;DM | 18 | no&nbsp;credits&nbsp;taken/all&nbsp;credits&nbsp;paid&nbsp;back&nbsp;duly | business | 3104 | <&nbsp;100&nbsp;DM | 4&nbsp;<=&nbsp;...&nbsp;<&nbsp;7&nbsp;years | 3 | male:&nbsp;single | none | 1 | building&nbsp;society&nbsp;savings&nbsp;agreement/life&nbsp;insurance | 31 | bank | own | 1 | skilled&nbsp;employee/official | 1 | yes,&nbsp;registered&nbsp;under&nbsp;the&nbsp;customers&nbsp;name | yes |
    | Adversarial&nbsp;sample | *no&nbsp;checking&nbsp;account | 18 | no&nbsp;credits&nbsp;taken/all&nbsp;credits&nbsp;paid&nbsp;back&nbsp;duly | *car&nbsp;(used) | 3104 | <&nbsp;100&nbsp;DM | 4&nbsp;<=&nbsp;...&nbsp;<&nbsp;7&nbsp;years | 3 | male:&nbsp;single | none | 1 | building&nbsp;society&nbsp;savings&nbsp;agreement/life&nbsp;insurance | 31 | bank | own | 1 | skilled&nbsp;employee/official | 1 | yes,&nbsp;registered&nbsp;under&nbsp;the&nbsp;customers&nbsp;name | yes |

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


