简体中文 | [English](./README.md)

# AdvBox

对抗样本是深度学习领域的一个重要问题，比如在图像上叠加肉眼难以识别的修改，就可以欺骗主流的深度学习图像模型，产生分类错误，指鹿为马，或者无中生有。这些问题对于特定领域（比如无人车、人脸识别）会产生严重的后果。因此AI模型对抗攻击及防御技术引起机器学习和安全领域的研究者及开发者越来越多的关注。
AdvBox( Adversarialbox ) 是一款由百度安全实验室研发，支持Paddle的AI模型安全工具箱。AdvBox集成了多种攻击算法，可以高效的构造对抗样本，进行模型鲁棒性评估或对抗训练，提高模型的安全性。它能为工程师、研究者研究模型的安全性提供便利，减少重复造轮子的精力与时间消耗。

---

## 名词解释

* 白盒攻击：攻击者可以知道模型的内部结构，训练参数，训练和防御方法等。
* 黑盒攻击：攻击者对攻击的模型的内部结构，训练参数等一无所知，只能通过输出与模型进行交互。
* 非定向攻击：攻击者只需要让目标模型对样本分类错误即可，但并不指定分类错成哪一类。
* 定向攻击：攻击者指定某一类，使得目标模型不仅对样本分类错误并且需要错成指定的类别。

---
## 攻击算法列表 

白盒攻击算法
+ FGSM
+ MI-FGSM
+ PGD
+ BIM
+ ILCM
+ C/W

黑盒攻击算法
+ SinglePixelAttack
+ TransferAttack

---
### 黑盒攻击示例

在mnist数据集，针对自己训练的CNN模型生成对抗样本。首先生成需要攻击的模型：    

    cd PaddleShield/Advbox/examples/image_cls
    pytho mnist_cnn_bapi.py


如果已有paddle2训练好的模型，不指定参数为非定向攻击可直接运行:

    python mnist_tutorial_singlepixelattack.py    

对于定向攻击，可指定目标类别，例如设置target为9（可为0-9任意值）    

    python mnist_tutorial_singlepixelattack.py  --target=9 

```shell
2021-04-25 13:51:26,187 - INFO - Attack location x=19 y=25
attack success, original_label=9, adversarial_label=2, count=17
2021-04-25 13:51:26,386 - INFO - Attack location x=1 y=6
attack success, original_label=7, adversarial_label=3, count=18
2021-04-25 13:51:26,587 - INFO - Attack location x=5 y=19
attack success, original_label=3, adversarial_label=8, count=19
2021-04-25 13:51:26,788 - INFO - Attack location x=20 y=20
attack success, original_label=4, adversarial_label=1, count=20
[TEST_DATASET]: fooling_count=20, total_count=20, fooling_rate=1.000000
SinglePixelAttack attack done
```

**Single Pixel Attack** 

<img src="./examples/image_cls/output/show/number5_adv.png" style="zoom:20%;" />

**Transfer Attack** 

迁移攻击的两种实现方式，分别用并行和串行。  

    python weighted_ensemble_attack_fgsm.py --target=330
    python serial_ensemble_attack_fgsm.py --target=1

类别282的小猫，经过黑盒攻击后被误识别为类别1金鱼。
<img src="./examples/image_cls/output/show/serial_ensemble_fgsm_diff_1.png" style="zoom:60%;" />
### 白盒攻击示例

以FGSM为例，其他攻击方法使用方式类似。采用imagenet数据集，vgg16的预训练模型作为攻击对象。

#### 1.FGSM非定向攻击

    cd PaddleShield/Advbox/examples/image_cls
    python imagenet_tutorial_fgsm.py

``` shell
label=717
input img shape:  [3, 224, 224]
attack success, adversarial_label=803
diff shape:  (224, 224, 3)
fgsm attack done
```
攻击成功，模型对于此图片的识别，label为717识别成label 803。

**FGSM untargeted attack**      
<img src="./examples/image_cls/output/show/fgsm_untarget_803.png" style="zoom:60%;" />

#### 2.FGSM定向攻击
定向攻击类别为266   

    python imagenet_tutorial_fgsm.py --target=266   

``` shell
label=717
input img shape:  [3, 224, 224]
attack success, adversarial_label=999
diff shape:  (224, 224, 3)
fgsm attack done
```

**FGSM targeted attack**
<img src="./examples/image_cls/output/show/fgsm_target_999.png" style="zoom:60%;" />

### 其他攻击方法示例结果

**PGD定向攻击**
<img src="./examples/image_cls/output/show/pgd_adv.png" style="zoom:20%;" />

**CW定向攻击**
<img src="./examples/image_cls/output/show/cw_adv.png" style="zoom:20%;" />

**BIM非定向攻击**

<img src="./examples/image_cls/output/show/bim_untarget_368.png" style="zoom:40%;" />