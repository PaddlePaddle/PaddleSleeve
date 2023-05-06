English | **[简体中文](/AdvBox/tabular_data/README_cn.md)**

# Advbox - Tabular Data
The Advbox-tabular_data is a sub-module of AdvBox used to generate adversarial samples for machine learning models with structured data.

## Installation
### Requirements
- Python >= 3.7
- numpy
- scikit-learn
- xgboost
- pandas
- tqdm

## Get Started with Advbox-tabular_data

These examples train a XGboost model on the german credit dataset and creates adversarial examples using the Advbox-tabular_data.  Here we trained the XGBoost model to feed to the Advbox-tabular_data predictor, it would also be possible to provide your own pretrained models to the Advbox-tabular_data predictor or uesd your own predictor. The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.

### Dataset
Our examples are based on the **[Statlog (German Credit Data) Data Set](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))**. We have downloaded the data to **[data/German_Credit_Data](/AdvBox/tabular_data/data/German_Credit_Data)**.

The German Credit Data with the following structure:

```
German_Credit_Data
|_ german.doc
|_ german.data
|_ german.data-numeric
```
 
### Examples
All examples are in the **[examples](/AdvBox/tabular_data/examples)** folder.

```
cd examples
```

- **[attack_german_credit_data_xgboost.py](/AdvBox/tabular_data/examples/attack_german_credit_data_xgboost.py)** demonstrates a example of classification task adversarial attack, train the XBGoost model using the German Credit Data and then use the Advbox-tabular_data to build the predictor and generate adversarial samples based on the attack Settings.
  - **Command-line parameters**
    - `--seed`: Random seed, default: 666.
    - `--data_path`: Path of file "german.data" in German Credit Data.
  - **Usage**
    ```
    python attack_german_credit_data_xgboost.py --seed 666 --data_path ../data/German_Credit_Data/german.data
    ```
  - **Result**

    | Samples | Status&nbsp;of&nbsp;existing&nbsp;checking&nbsp;account | Duration&nbsp;in&nbsp;month | Credit&nbsp;history | Purpose | Credit&nbsp;amount | Savings&nbsp;account/bonds | Present&nbsp;employment&nbsp;since | Installment&nbsp;rate&nbsp;in&nbsp;percentage&nbsp;of&nbsp;disposable&nbsp;income | Personal&nbsp;status&nbsp;and&nbsp;sex | Other&nbsp;debtors/guarantors | Present&nbsp;residence&nbsp;since | Property | Age&nbsp;in&nbsp;years | Other&nbsp;installment&nbsp;plans | Housing | Number&nbsp;of&nbsp;existing&nbsp;credits&nbsp;at&nbsp;this&nbsp;bank | Job | Number&nbsp;of&nbsp;people&nbsp;being&nbsp;liable&nbsp;to&nbsp;provide&nbsp;maintenance&nbsp;for | Telephone | foreign&nbsp;worker |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | Original&nbsp;sample | <&nbsp;0&nbsp;DM | 18 | no&nbsp;credits&nbsp;taken/all&nbsp;credits&nbsp;paid&nbsp;back&nbsp;duly | business | 3104 | <&nbsp;100&nbsp;DM | 4&nbsp;<=&nbsp;...&nbsp;<&nbsp;7&nbsp;years | 3 | male:&nbsp;single | none | 1 | building&nbsp;society&nbsp;savings&nbsp;agreement/life&nbsp;insurance | 31 | bank | own | 1 | skilled&nbsp;employee/official | 1 | yes,&nbsp;registered&nbsp;under&nbsp;the&nbsp;customers&nbsp;name | yes |
    | Adversarial&nbsp;sample | *no&nbsp;checking&nbsp;account | 18 | no&nbsp;credits&nbsp;taken/all&nbsp;credits&nbsp;paid&nbsp;back&nbsp;duly | *car&nbsp;(used) | 3104 | <&nbsp;100&nbsp;DM | 4&nbsp;<=&nbsp;...&nbsp;<&nbsp;7&nbsp;years | 3 | male:&nbsp;single | none | 1 | building&nbsp;society&nbsp;savings&nbsp;agreement/life&nbsp;insurance | 31 | bank | own | 1 | skilled&nbsp;employee/official | 1 | yes,&nbsp;registered&nbsp;under&nbsp;the&nbsp;customers&nbsp;name | yes |

- **[attack_german_credit_data_xgboost_regression.py](/AdvBox/tabular_data/examples/attack_german_credit_data_xgboost_regression.py)** demonstrates a example ofregression task adversarial attack, train the XBGoost model using the German Credit Data and then use the Advbox-tabular_data to build the predictor and generate adversarial samples based on the attack Settings.
  - **Command-line parameters**
    - `--seed`: Random seed, default: 666.
    - `--data_path`: Path of file "german.data" in German Credit Data.
  - **Usage**
    ```
    python attack_german_credit_data_xgboost_regression.py --seed 666 --data_path ../data/German_Credit_Data/german.data
    ```
  - **Result**

    | Samples | Status&nbsp;of&nbsp;existing&nbsp;checking&nbsp;account | Duration&nbsp;in&nbsp;month | Credit&nbsp;history | Purpose | Credit&nbsp;amount | Savings&nbsp;account/bonds | Present&nbsp;employment&nbsp;since | Installment&nbsp;rate&nbsp;in&nbsp;percentage&nbsp;of&nbsp;disposable&nbsp;income | Personal&nbsp;status&nbsp;and&nbsp;sex | Other&nbsp;debtors/guarantors | Present&nbsp;residence&nbsp;since | Property | Age&nbsp;in&nbsp;years | Other&nbsp;installment&nbsp;plans | Housing | Number&nbsp;of&nbsp;existing&nbsp;credits&nbsp;at&nbsp;this&nbsp;bank | Job | Number&nbsp;of&nbsp;people&nbsp;being&nbsp;liable&nbsp;to&nbsp;provide&nbsp;maintenance&nbsp;for | Telephone | foreign&nbsp;worker |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | Original&nbsp;sample | 0&nbsp;<=&nbsp;...&nbsp;<&nbsp;200&nbsp;DM | 36 | delay&nbsp;in&nbsp;paying&nbsp;off&nbsp;in&nbsp;the&nbsp;past | business | 4455 | <&nbsp;100&nbsp;DM | 1&nbsp;<=&nbsp;...&nbsp;<&nbsp;4&nbsp;years | 2 | male&nbsp;:&nbsp;divorced/separated | none | 2 | building&nbsp;society&nbsp;savings&nbsp;agreement/life&nbsp;insurance | 30 | stores | own | 2 | management/self-employed/highly&nbsp;qualified&nbsp;employee/&nbsp;officer | 1 | yes,&nbsp;registered&nbsp;under&nbsp;the&nbsp;customers&nbsp;name | yes |
    | Adversarial&nbsp;sample | 0&nbsp;<=&nbsp;...&nbsp;<&nbsp;200&nbsp;DM | 36 | delay&nbsp;in&nbsp;paying&nbsp;off&nbsp;in&nbsp;the&nbsp;past | car&nbsp;(used) | 4455 | <&nbsp;100&nbsp;DM | unemployed | 2 | male&nbsp;:&nbsp;divorced/separated | none | 1 | building&nbsp;society&nbsp;savings&nbsp;agreement/life&nbsp;insurance | 30 | stores | own | 1 | management/self-employed/highly&nbsp;qualified&nbsp;employee/officer | 1 | yes,&nbsp;registered&nbsp;under&nbsp;the&nbsp;customers&nbsp;name | yes |


## Citing

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
