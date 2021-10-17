简体中文 | [English](./README.md)

# AdvBox

对抗样本是深度学习领域的一个重要问题，比如在图像上叠加肉眼难以识别的修改，就可以欺骗主流的深度学习图像模型，产生分类错误，指鹿为马，或者无中生有。这些问题对于特定领域（比如无人车、人脸识别）会产生严重的后果。因此AI模型对抗攻击及防御技术引起机器学习和安全领域的研究者及开发者越来越多的关注。对于对抗样本的研究可以找出当前机器学习算法的局限性和潜在威胁，提供鲁棒性的衡量工具，有助于寻找提升模型鲁棒性的方法。
AdvBox( Adversarialbox ) 是一款由百度安全实验室研发，支持Paddle的AI模型安全工具箱。AdvBox集成了多种攻击算法，可以高效的构造对抗样本，进行模型鲁棒性评估或对抗训练，提高模型的安全性。它能为工程师、研究者研究模型的安全性提供便利，减少重复造轮子的精力与时间消耗。

---

## 名词解释

* 白盒攻击：攻击者可以知道模型的内部结构，训练参数，训练和防御方法等。
* 黑盒攻击：攻击者对攻击的模型的内部结构，训练参数等一无所知，只能通过输出与模型进行交互。
* 非定向攻击：攻击者只需要让目标模型对样本分类错误即可，但并不指定分类错成哪一类。
* 定向攻击：攻击者指定某一类，使得目标模型不仅对样本分类错误并且需要错成指定的类别。

---
## 攻击算法列表

| Adversarial Attack Methods                                    | White-Box | Black-Box | Ensemble  |  AdvTrain   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--:|:--:|:--:|:--:|
| [FGSM (FastGradientSignMethodAttack)](attacks/gradient_method.py)                | ✓  |   | ✓ | ✓ |
| [FGSMT (FastGradientSignMethodTargetedAttack)](attacks/gradient_method.py)       | ✓  |   | ✓ | ✓ |
| [BIM (BasicIterativeMethodAttack)](attacks/gradient_method.py)                   | ✓  |   | ✓ | ✓ |
| [ILCM (IterativeLeastLikelyClassMethodAttack)](attacks/gradient_method.py)       | ✓  |   | ✓ | ✓ |
| [MI-FGSM (MomentumIteratorAttack)](attacks/gradient_method.py)                   | ✓  |   | ✓ | ✓ |
| [PGD (ProjectedGradientDescentAttack)](attacks/gradient_method.py)               | ✓  |   | ✓ | ✓ |
| [CW_L2 (CWL2Attack)](attacks/cw.py)                                              | ✓  |   |   | ✓ |
| [SinglePixelAttack](attacks/single_pixel_attack.py)                              |    | ✓ |   |   |
| [HopSkipJumpAttack](attacks/hop_skip_jump_attack.py)                             |    | ✓ |   |   |

---
### 黑盒攻击示例

在mnist数据集，针对自己训练的CNN模型生成对抗样本。首先生成需要攻击的模型：    

    cd PaddleShield/Advbox/examples/image_cls
    python mnist_cnn_bapi.py


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

<img src="./examples/image_cls/output/show/pgd_adv.png" style="zoom:20%;" />

<img src="./examples/image_cls/output/show/cw_adv.png" style="zoom:20%;" />

**BIM非定向攻击**

<img src="./examples/image_cls/output/show/bim_untarget_368.png" style="zoom:40%;" />

### 利用AdvBox生成一个对抗样本

```python
import sys
sys.path.append("..")
import paddle
import numpy as np
from adversary import Adversary
from attacks.cw import CW_L2
from models.whitebox import PaddleWhiteBoxModel

from classifier.towernet import transform_eval, TowerNet, MEAN, STD
model_0 = TowerNet(3, 10, wide_scale=1)
model_1 = TowerNet(3, 10, wide_scale=2)

advbox_model = PaddleWhiteBoxModel(
    [model_0, model_1],
    [1, 1.8],
    (0, 1),
    mean=MEAN,
    std=STD,
    input_channel_axis=0,
    input_shape=(3, 256, 256),
    loss=paddle.nn.CrossEntropyLoss(),
    nb_classes=10)

# init attack with the ensembled model
attack = CW_L2(advbox_model)

cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
test_loader = paddle.io.DataLoader(cifar10_test, batch_size=1)

data = test_loader().next()
img = data[0][0]
label = data[1]

# init adversary status
adversary = Adversary(img.numpy(), int(label))
target = np.random.randint(advbox_model.num_classes())
while label == target:
    target = np.random.randint(advbox_model.num_classes())
print(label, target)
adversary.set_status(is_targeted_attack=True, target_label=target)

# launch attack
adversary = attack(adversary, attack_iterations=50, verbose=True)

if adversary.is_successful():
    original_img = adversary.original
    adversarial_img = adversary.adversarial_example
    print("Attack succeeded.")
else:
    print("Attack failed.")
```

# 对抗训练

## AdvBox对抗训练(advtraining)提供:
- 基于主流攻击算法 **[FGSM/PGD/BIM/ILCM/MI-FGSM](#AdvBox/attacks)** 的数据增强工具，用于对抗训练
- 紧凑便捷的对抗训练工具API：
    + 支持将训练数据按照比例进行对抗扰动，便于接入已有的paddle分类模型训练流程
    + 支持事先按照设定权重，进行模型融合的对抗样本生成
    + 支持多对抗攻击方法的对抗样本生成
- Advtraining **[tutorial scripts](#AdvBox/examples/image_adversarial_training)** 演示脚本，基于Cifar10和Mini-ImageNet数据集

## 如何运行对抗训练演示
对抗训练演示包含以下实验：
- 基于Preactresnet在Cifar10和Mini-ImageNet的对抗训练Benchmark
- 基于Towernet在Mini-ImageNet数据集上使用PGD数据增强的微调实验
- 附加的未完成的实验

运行以下命令来运行演示
1. `cd AdvBox/examples/image_adversarial_training`
2. `python run_advtrain_main.py`
3. `python model_evaluation_tutorial.py`

**PreactResnet在不同对抗训练设定下的鲁棒性表现**

| Evaluation-Method | Mini-ImageNet-FGSM | Mini-ImageNet-PGD-20 |
| :----: | :----: | :----: |
|   val_acc: _ / natural_acc: _ / fooling_rate: _   |   preactresnet   |   preactresnet   |
|   Natural Adversarial Training(p=0.1, fgsm(default))   |   0.980 / 0.986 / 0.282   |   0.980 / 0.986 / 0.984   |
|   Natural Adversarial Training(p=0.1, PGD(default))   |   0.983 / 0.978 / 0.098   |   0.983 / 0.982 /0.850   |
|  TRADES(beta=1.0, fgsm(default))  |  0.989 / 0.994 / 0.146  |  0.989 / 0.994 / 0.956  |
|  TRADES(beta=1.0, PGD(default))  |  0.990 / 0.992 / 0.028  |  0.990 / 0.996 / 0.540  |
|  TRADES(beta=1.0, LD(default))  |  0.990 / 0.996 / 0.020  |  0.990 / 0.992 / 0.734  |

如表所示，对抗训练可以在牺牲很少精度的情况下，增加模型的鲁棒性。

## 对抗训练的helloword
```python
import sys
sys.path.append("..")
import paddle
from attacks.gradient_method import FGSM, PGD
from attacks.cw import CW_L2
from models.whitebox import PaddleWhiteBoxModel
from defences.adversarial_transform import ClassificationAdversarialTransform

from classifier.towernet import transform_train, TowerNet, MEAN, STD
model_0 = TowerNet(3, 10, wide_scale=1)
model_1 = TowerNet(3, 10, wide_scale=2)

advbox_model = PaddleWhiteBoxModel(
    [model_0, model_1],
    [1, 1.8],
    (0, 1),
    mean=MEAN,
    std=STD,
    input_channel_axis=0,
    input_shape=(3, 256, 256),
    loss=paddle.nn.CrossEntropyLoss(),
    nb_classes=10)

# "p" controls the probability of this enhance.
# for base model training, we set "p" == 0, so we skipped adv trans data augmentation.
# for adv trained model, we set "p" == 0.05, which means each batch
# will probably contain 5% adv trans augmented data.
enhance_config1 = {"p": 0.1}
enhance_config2 = {"p": 0.1}
init_config3 = {"norm": 'L2', "epsilon_ball": 8/255, "epsilon_stepsize": 2/255}
enhance_config3 = {"p": 0.05,
                   "attack_iterations": 15,
                   "c_search_steps": 6,
                   "verbose": False}

adversarial_trans = ClassificationAdversarialTransform(advbox_model,
                                                       [FGSM, PGD, CW_L2],
                                                       [None, None, init_config3],
                                                       [enhance_config1, enhance_config2, enhance_config3])

cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
train_loader = paddle.io.DataLoader(cifar10_train, batch_size=16)

for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = paddle.unsqueeze(data[1], 1)
    x_data_augmented, y_data_augmented = adversarial_trans(x_data.numpy(), y_data.numpy())
```

# 目标检测器的对抗扰动
目标检测器的对抗扰动主要用于目标检测器的对抗训练和鲁棒性测评，主要分为电子世界下和物理世界下的对抗扰动。
这里我们提供一种电子世界下对PP-YOLO目标检测器扰动的演示。该演示基于 **[PaddleDetection](#https://github.com/PaddlePaddle/PaddleDetection)** 项目。

**用于Feed & Sniff的图像**

<table>
<tr>
    <td align="center"><img src="./examples/objectdetector/dataloader/demo_pics/000000014439.jpg" width=300></td>
    <td align="center"><img src="./examples/objectdetector/dataloader/demo_pics/masked_0014439.png" width=300></td>
</tr>

<tr>
    <td align="center">Original Image</td>
    <td align="center">Masked Image</td>
</tr>
</table>

在`PaddleSleeve/AdvBox/examples/objectdetector`，我们展示了一种称之为目标消失攻击的目标检测器
对抗方法。该演示是在可以获取模型权重信息的情况下，用受害图和制作的目标图获得关键张量，用PGD方法迭代更新扰动图
像，使受害图和目标图对应的分类置信度张量的KL散度最小。该演示中，我们成功的使被扰动后的图片`000000014439.jpg`，
在PP-YOLO下，对风筝这个大目标造成了漏检。

- 友情提示：由于PaddlePaddle<=2.1的版本，暂时不支持对于`paddle.nn.SyncBatchNorm`在eval()模式下
的反向传播功能，我们需要将所有的`sync-bn`组件置换喂`bn`组件(因为`paddle.nn.BatchNorm`支持eval()
模式的求导).
 
想要为其他目标检测器定制攻击脚本，可以参照以下方法置换`paddle.nn.SyncBatchNorm`：

- 如目标检测器配置文件类似于`configs/yolov3/_base_/yolov3_darknet53.yml`，在第三行添加`norm_type: bn`
- 如目标检测器配置文件类似于`configs/ppyolo/ppyolo_mbv3_large_coco.yml`，在第九行添加`norm_type: bn`

## 运行目标消失演示
在把所有`sync-bn`组件置换为`bn`组件后，运行以下命令：
1. `cd PaddleSleeve/AdvBox/examples/objectdetector`
2. `python target_ghosting_demo.py -c configs/ppyolo/ppyolo_mbv3_large_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams --infer_img=dataloader/demo_pics/000000014439.jpg --target_img=dataloader/demo_pics/masked_0014439.png`

成功的运行`target_ghosting_demo.py`可以产生以下图片：

**图片对比**

<table align="center">
<tr>
    <td align="center"><img src="./examples/objectdetector/output/out_000000014439.jpg" width=300></td>
    <td align="center"><img src="./examples/objectdetector/output/out_masked_0014439.png" width=300></td>
    <td align="center"><img src="./examples/objectdetector/output/out_adv_000000014439.jpg.png" width=300></td>
</tr>

<tr>
    <td align="center">Original Image Detection Result</td>
    <td align="center">Masked Image Detection Result</td>
    <td align="center">Adv Image Detection Result</td>
</tr>
</table>


# 对抗样本去噪算法列表

基本去噪算法

## 对抗样本去噪示例

- [基本去噪方法](#AdvBox/denoisers)
    + 高斯滤波（Gaussian Filter）
    + 中值滤波（Median Filter）
    + 均值滤波（Mean Filter）
    + 方框滤波（Box Filter）
    + 双边滤波（Bilateral Filter）
    + 像素偏移（Pixel Deflection）
    + JPEG压缩
    + DCT压缩
    + PCA降维
    + 高斯噪声 （GaussianNoise）
    + 椒盐噪声 （SaltPepperNoise）
    + 随机缩放填充
- 在一幅图像上使用FGSM攻击并去噪 **[tutorial python script](#AdvBox/examples/mini_imagenet_evaluation_tool.py)**.
- **命令行参数介绍**
  - `--image_path`  
  : 要处理的图像路径，用户可以上传图像到文件夹：AdvBox/examples/image_cls/input。我们提供了一些采集自mini-imagenet数据集的图像样本：
    + input/schoolbus.png
    + input/vase.png
    + input/lion.png
    + input/hourglass.png
    + input/crate.png
    + input/malamute.png
  - `--method`  
  : 去噪方法的名称，如下：
    + GaussianBlur
    + MedianBlur
    + MeanFilter
    + BoxFilter
    + BilateralFilter
    + PixelDeflection
    + JPEGCompression
    + DCTCompress
    + PCACompress
    + GaussianNoise
    + SaltPepperNoise
    + ResizePadding

  - 在Mini-ImageNet数据集上使用FGSM攻击图像并去噪 **[tutorial python script](#AdvBox/examples/imagenet_tutorial_fgsm_denoise.py)**.
  - **命令行参数介绍**
    - `--dataset_path`  
    : 要处理的mini-imagenet数据集（.pkl）路径，可以将数据集下载至：AdvBox/examples/image_cls/input中。
    - `--label_path`  
    : 要处理的数据集对应的类别标签，可以将文件放在：AdvBox/examples/image_cls/input。我们提供了测试集的标签：
      + input/mini_imagenet_test_labels.txt
    - `--mode`
    : 数据集类型, 'train'，'test'，或者 'val'。默认是 Default 'test'.
    - `--method`  
    : 去噪方法的名称，如下：
      + GaussianBlur
      + MedianBlur
      + MeanFilter
      + BoxFilter
      + BilateralFilter
      + PixelDeflection
      + JPEGCompression
      + DCTCompress
      + PCACompress
      + GaussianNoise
      + SaltPepperNoise
      + ResizePadding

## 去噪算法使用示例
在单幅图像或者mini-imagenet数据集上对清晰图像或者对抗样本使用去噪算法。

### 单幅图像去噪示例
给定一幅图像，首先使用FGSM方法产生对抗样本（AE），在使用去噪算法对AE进行去噪，同时对比对输入的清晰图像的去噪结果。

#### 执行代码：
```shell
cd PaddleShield/Advbox/examples/image_cls
python imagenet_tutorial_fgsm_denoise.py --method='GaussianBlur' --image_path='input/schoolbus.png'
```

#### 输出结果：
```
input image label: school bus
input image shape:  [3, 84, 84]
FGSM attack succeeded, adversarial_label: rubber eraser, rubber, pencil eraser
FGSM attack done
GaussianBlur denoise succeeded
GaussianBlur denoise doesn't change the label of the input image
GaussianBlur denoise done
```

#### 结果解读：
```
1. 原始模型将输入图像识别为：school bus；  
2. FGSM攻击输入图像，得到对抗样本，模型将该对抗样本识别为：rubber eraser, rubber, pencil eraser；  
3. 去噪算法对对抗样本进行去噪，得到去噪结果，模型将该结果识别为：school bus。```
```

#### 可视化结果
<div align=center>
<img src="./examples/image_cls/output/GaussianBlur_Denoising_Comparison.png" style="zoom:40%;"/>
</div>

#### 其他去噪方法示例
**中值滤波**
```shell
python imagenet_tutorial_fgsm_denoise.py --method='MedianBlur' --image_path='input/vase.png'
```
<div align=center>
<img src="./examples/image_cls/output/MedianBlur_Denoising_Comparison.png" style="zoom:40%;" />
</div><br/>

**均值滤波**
```shell
python imagenet_tutorial_fgsm_denoise.py --method='MeanFilter' --image_path='input/lion.png'
```

<div align=center>
<img src="./examples/image_cls/output/MeanFilter_Denoising_Comparison.png" style="zoom:40%;" />
</div><br/>

**方框滤波**
```shell
python imagenet_tutorial_fgsm_denoise.py --method='BoxFilter' --image_path='input/hourglass.png'
```
<div align=center>
<img src="./examples/image_cls/output/BoxFilter_Denoising_Comparison.png" style="zoom:40%;" />
</div><br/>

**双边滤波**
```shell
python imagenet_tutorial_fgsm_denoise.py --method='BilateralFilter' --image_path='input/crate.png'
```
<div align=center>
<img src="./examples/image_cls/output/BilateralFilter_Denoising_Comparison.png" style="zoom:40%;" />
</div><br/>

**像素偏移**
```shell
python imagenet_tutorial_fgsm_denoise.py --method='PixelDeflection' --image_path='input/malamute.png'
```
<div align=center>
<img src="./examples/image_cls/output/PixelDeflection_Denoising_Comparison.png" style="zoom:40%;" />
</div><br/>

**JPEG压缩**
```shell
python imagenet_tutorial_fgsm_denoise.py --method='JPEGCompression' --image_path='input/schoolbus.png'
```
<div align=center>
<img src="./examples/image_cls/output/JPEGCompression_Denoising_Comparison.png" style="zoom:40%;" />
</div><br/>

### Mini-ImageNet数据集去噪示例
给定mini-imagenet数据集，依次对数据集中的每幅图像：先使用FGSM方法产生对抗样本（AE），在使用去噪算法对AE进行去噪，同时对比对输入的清晰图像的去噪结果。

#### 执行代码:
```shell
cd PaddleShield/Advbox/examples/image_cls
python mini_imagenet_evaluation_tool.py --method='GaussianBlur' --dataset_path='input/mini-imagenet-cache-test.pkl' --label_path='mini_imagenet_test_labels.txt'
```

#### 输出结果:
```
100%|█████| 12000/12000 [2:45:59<00:00,  1.20it/s, ORI_ACC=0.439, AE_ACC=0.000, DE_AE_ACC=0.063, DE_ORI_ACC=0.010]
```

#### 定量结果 (分类准确率):
| 去噪方法 | 清晰图像 | 对抗样本 | 对抗样本去噪 | 清晰图像去噪 |
|:-|:-:|:-:|:-:|:-:|
| 高斯滤波    | 43.9%  | 0.0%  | 6.3% | 10.0% |
| 中值滤波    | 43.9%  | 0.0%  | 7.2% | 10.4% |
| 均值滤波    | 43.9%  | 0.0%  | 5.8% | 9.0% |
| 方框滤波    | 43.9%  | 0.0%  | 7.4% | 14.4% |
| 双边滤波    | 43.9%  | 0.0%  | 5.8% | 9.0% |
| 像素偏移    | 43.9%  | 0.0%  | 11.7% | 18.3% |
| JPEG压缩    | 43.9%  | 0.0%  | 12.6% | 19.5% |
| DCT压缩     | 43.9%  | 0.0%  | 10.9% | 16.5% |
| PCA降维     | 43.9%  | 0.0%  | 11.7% | 20.6% |
| 高斯噪声    | 43.9%  | 0.0%  | 8.0% | 10.0% |
| 椒盐噪声    | 43.9%  | 0.0%  | 7.3% | 11.0% |
| 随机缩放填充 | 43.9%  | 0.0%  | 18.9% | 22.5% |
