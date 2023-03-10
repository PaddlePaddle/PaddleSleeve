English | **[简体中文](/AdvBox/tabular_adversarial_toolbox/README_cn.md)**

# Advbox - Tabular Adversarial Toolbox
The tabular adversarial toolbox(TAT) is a sub-module of AdvBox used to generate adversarial samples for structured data training models.

## Installation
### Requirements
- Python >= 3.7
- numpy
- scikit-learn
- xgboost
- pandas

## Get Started with Tabular Adversarial Toolbox

These examples train a XGboost model on the german credit dataset and creates adversarial examples using the TAT.  Here we trained the XGBoost model to feed to the TAT predictor, it would also be possible to provide your own pretrained models to the TAT predictor or uesd your own predictor. The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.

### Dataset
Our examples are based on the **[Statlog (German Credit Data) Data Set](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))**. We have downloaded the data to **[data/German_Credit_Data](/AdvBox/tabular_adversarial_toolbox/data/German_Credit_Data)**.

The German Credit Data with the following structure:

```
German_Credit_Data
|_ german.doc
|_ german.data
|_ german.data-numeric
```
 
### Examples
All examples are in the **[examples](/AdvBox/tabular_adversarial_toolbox/examples)** folder.

```
cd examples
```

- **[attack_german_credit_data_xgboost.py](/AdvBox/tabular_adversarial_toolbox/examples/attack_german_credit_data_xgboost.py)** demonstrates a example of train the XBGoost model using the German Credit Data and then use TAT to build the predictor and generate adversarial samples based on the attack Settings.
  - **Command-line parameters**
    - `--seed`: Random seed, default: 666.
    - `--data_path`: Path of file "german.data" in German Credit Data.
  - **Usage**
    ```
    python attack_german_credit_data_xgboost.py --seed 666 --data_path ../data/German_Credit_Data/german.data
    ```
  - **Result**

    | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Status&nbsp;of&nbsp;existing&nbsp;checking&nbsp;account | Duration in month | Credit history | Purpose | Credit amount | Savings account/bonds | Present employment since | Installment rate in percentage of disposable income | Personal status and sex | Other debtors/guarantors | Present residence since | Property | Age in years | Other installment plans | Housing | Number of existing credits at this bank | Job | Number of people being liable to provide maintenance for | Telephone | foreign worker |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | Original sample    | < 0 DM | 18 | no credits taken/all credits paid back duly | business | 3104 | < 100 DM | 4 <= ... < 7 years | 3 | male : single | none | 1 | building society savings agreement/life insurance | 31 | bank | own | 1 | skilled employee/official | 1 | yes, registered under the customers name | yes |
    | Adversarial sample | *no checking account | 18 | no credits taken/all credits paid back duly | *car (used) | 3104 | < 100 DM | 4 <= ... < 7 years | 3 | male : single | none | 1 | building society savings agreement/life insurance | 31 | bank | own | 1 | skilled employee/official | 1 | yes, registered under the customers name | yes



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
