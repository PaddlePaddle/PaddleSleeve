English | [简体中文](./README_cn.md)

# AdvBox

AdvBox (Adversarialbox) is a Paddlepaddle Open Source project that provides users with a series of AI model security tools, including adversarial examples (AEs) generation techniques and model-based adversarial data augmentation. 

Since the existence of adversarial examples may be an inherent weakness of deep learning models, it is important to benchmark deep learning models and improve their robustness against AEs. The purpose of the AdvBox is to help users generate and use adversarial examples conveniently in Paddlepaddle.

The project also contains plenty of useful tutorials for different AI applications and scenarios.

(A command-line tool is given to generate adversarial examples with Zero-Coding which is inspired and based on FoolBox v1.)

# Attack Methods

**FGSM untargeted attack**      
<img src="./examples/image_cls/output/show/fgsm_untarget_803.png" style="zoom:60%;" />

**PGD targeted attack**
<img src="./examples/image_cls/output/show/pgd_adv.png" style="zoom:20%;" />

**CW targeted attack**
<img src="./examples/image_cls/output/show/cw_adv.png" style="zoom:20%;" />


## Table of Adversarial Attack Methods

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

 
## To generate an AE in AdvBox

```python
import sys
sys.path.append("..")
import paddle
import numpy as np
from adversary import Adversary
from attacks.gradient_method import FGSM
from attacks.cw import CW_L2
from models.whitebox import PaddleWhiteBoxModel

from classifier.definednet import transform_eval, TowerNet
model_0 = TowerNet(3, 10, wide_scale=1)
model_1 = TowerNet(3, 10, wide_scale=2)

# set fgsm attack configuration
fgsm_attack_config = {"norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
paddle_model = PaddleWhiteBoxModel(
    [model_0, model_1],   # ensemble two models
    [1, 1.8],             # dictate weight
    paddle.nn.CrossEntropyLoss(),
    (-3, 3),
    channel_axis=3,
    nb_classes=10)

# FGSM attack, init attack with the ensembled model
# attack = FGSM(paddle_model)
attack = CW_L2(paddle_model, learning_rate=0.01)

cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
test_loader = paddle.io.DataLoader(cifar10_test, batch_size=1)

data = test_loader().next()
img = data[0][0]
label = data[1]

# init adversary status
adversary = Adversary(img.numpy(), int(label))
target = np.random.randint(paddle_model.num_classes())
while label == target:
    target = np.random.randint(paddle_model.num_classes())
print(label, target)
adversary.set_status(is_targeted_attack=True, target_label=target)

# launch attack
# adversary = attack(adversary, **fgsm_attack_config)
adversary = attack(adversary, attack_iterations=100, verbose=True)

if adversary.is_successful():
    original_img = adversary.original
    adversarial_img = adversary.adversarial_example
    print("Attack succeeded.")
else:
    print("Attack failed.")
```


# Adversarial Training

## AdvBox Adversarial Training(defences) provides:

- Mainstream attack methods **[FGSM/PGD/BIM/ILCM/MI-FGSM](#AdvBox/attacks)** for model adversarial training.
- A unified yet handy adversarial training API: 
    + AEs generation/transformation in data-flow style, which is easy to work with batch feeding in the training process.
    + Supports weighted model ensembling for AEs generation/transformation.
    + Supports multi-methods adversarial training.
    + Allows users to specify settings for each adversarial attack method, including their probabilities to take effect.
- A **[tutorial python script](#AdvBox/examples/cifar10_tutorial_fgsm_advtraining.py)** uses the Cifar10 dataset for adversarial training demonstration.

## Easy to use adversarial training 
```python
import sys
sys.path.append("..")
import numpy as np
import paddle
from attacks.gradient_method import FGSM, PGD
from attacks.cw import CW_L2
from models.whitebox import PaddleWhiteBoxModel
from defences.adversarial_transform import ClassificationAdversarialTransform

from classifier.definednet import transform_train, TowerNet
model_0 = TowerNet(3, 10, wide_scale=1)
model_1 = TowerNet(3, 10, wide_scale=2)

# set fgsm attack configuration
fgsm_attack_config = {"norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
paddle_model = PaddleWhiteBoxModel(
    [model_0, model_1],  # ensemble two models
    [1, 1.8],  # dictate weight
    paddle.nn.CrossEntropyLoss(),
    (-3, 3),
    channel_axis=3,
    nb_classes=10)

# "p" controls the probability of this enhance.
# for base model training, we set "p" == 0, so we skipped adv trans data augmentation.
# for adv trained model, we set "p" == 0.05, which means each batch
# will probably contain 5% adv trans augmented data.
enhance_config = {"p": 0.1, "norm_ord": np.inf, "epsilons": 0.0005, "epsilon_steps": 1, "steps": 1}
enhance_config2 = {"p": 0.1, "norm_ord": np.inf, "epsilons": 0.001, "epsilon_steps": 3, "steps": 3}
init_config3 = {"learning_rate": 0.01}
enhance_config3 = {"p": 0.05,
                   "attack_iterations": 15,
                   "c_search_steps": 6,
                   "verbose": False}

adversarial_trans = ClassificationAdversarialTransform(paddle_model,
                                                       [FGSM, PGD, CW_L2],
                                                       [None, None, init_config3],
                                                       [enhance_config, enhance_config2, enhance_config3])

cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
train_loader = paddle.io.DataLoader(cifar10_train, batch_size=16)

for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = paddle.unsqueeze(data[1], 1)
    x_data_augmented, y_data_augmented = adversarial_trans(x_data.numpy(), y_data.numpy())
    print(batch_id)
```
# Contributing
We appreciate your contributions!

# Citing
If you find this toolbox useful for your research, please consider citing.
