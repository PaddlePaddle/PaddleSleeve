English | [简体中文](./README_cn.md)

# Robustness
Robustness is a robustness benchmark tool for vision DNN models on PaddlePaddle, PyTorch, etc.
It inherits the design from foolbox, and is designed to be agnostic to the deep learning frameworks the models are built on.

## Robustness benchmark
### various Safety scenarios
- GaussianNoise
- UniformNoise
- GaussianBlur
- Brightness
- ContrastReduction
- MotionBlur
- Rotation
- SaltAndPepper
- Spatial
- Fog
- Frost
- Snow

### various metrics
- MeanAbsoluteError
- L_inf
- L0
- L2

### different frameworks

- Paddle
- Pytorch

## 1. Installation

```shell
cd paddleshield/Robustness

# Create and activate conda virtual environment (not required)
conda create -n perce python=3.7
conda activate perce

# Install the required libraries for this project
pip install -e .

# If you want to use the paddle model, you need to install paddle. Note: Install according to the paddle quick installation instructions below.
# The paddle version of this project is 2.1.1
python -m pip install paddlepaddle-gpu==2.1.1.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# If you want to use the paddlehub model, you need to install paddlehub. Note: Install according to the paddlehub quick installation instructions below.
# The paddlehub version of this project is 2.1.0
pip install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple
```

> [paddle quick install](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)
>
> [paddlehub quick install](https://www.paddlepaddle.org.cn/hub)
>
> Note: For the pytorch environment, users also need to install it by themselves.

## 2. Running

This project supports two calling methods, which can be called through command line parameters in the main program, or the corresponding script file can be run directly.

### 2.1 Command line parameter call

- **Introduction to command line parameters**

  - `--framework`：Choose deep learning framework, support paddle, paddlehub, pytorch, pytorchhub.
  - `--model`：Select the pre-trained model. Currently, the paddle framework supports resnet18, resnet50, vgg16, the paddlehub framework supports pre-trained models such as YOLOv3 and SSD, the pytorch framework supports pre-trained models such as vgg11 and resnet18, the pytorchhub framework supports pre-trained models such as YOLOv5.
  - `--image`：Users can upload their own pictures to the Robustness/perceptron/utils/images folder of the original pictures being attacked.
  - `--criteria`：Provides class to wrap all adversarial criterions so that attacks has uniform API access.

  ```python
  "criterions": [
    "misclassification",
    "confident_misclassification",
    "topk_misclassification",
    "target_class",
    "original_class_probability",
    "target_class_probability",
    "misclassification_antiporn",
    "misclassification_safesearch",
    "target_class_miss",
    "target_class_miss_google",
    "weighted_ap"
  ]
  ```

  - `--metric`：Provides different attack and evaluation approaches.

  ```python
  "metrics": [
    "additive_gaussian_noise",
    "additive_uniform_noise",
    "blend_uniform_noise",
    "gaussian_blur",
    "brightness",
    "contrast_reduction",
    "motion_blur",
    "rotation",
    "salt_and_pepper_noise",
    "spatial",
    "contrast",
    "horizontal_translation",
    "vertical_translation",
    "snow",
    "fog",
    "frost"
  ]
  ```

- **Examples**

```shell
# Use the paddle framework, use the picture Robustness/perceptron/utils/images/example.jpg Gaussian smooth attack pre-trained model restnet18

cd paddleshield/Robustness

# paddle：The results can be found under Robustness/examples/images.
python perceptron/launcher.py  --framework paddle --model resnet18 --criteria misclassification --metric gaussian_blur --image example.jpg

# Use other frameworks

# paddlehub: The results can be found under Robustness/images.
python perceptron/launcher.py  --framework paddlehub --model paddlehub_ssd_vgg16_300_coco2017 --criteria target_class_miss --metric gaussian_blur --image example.jpg --target_class -1

# pytorch:
python perceptron/launcher.py  --framework pytorch --model resnet18 --criteria misclassification --metric gaussian_blur --image example.jpg

# pytorchhub: The results can be found under Robustness/images.
python perceptron/launcher.py  --framework pytorchhub --model pytorchhub_yolov5s --criteria target_class_miss --metric gaussian_blur --image example.jpg --target_class -1
```

- **Results**

Image classification:

Paddle-ResNet18
<img src="./perceptron/utils/images/doc/console_gaussianblur_minivan2mobilehome.jpeg" style="zoom:60%;" />

<img src="./perceptron/utils/images/doc/pic_gaussianblur_minivan2mobilehome.jpeg" style="zoom:50%;" />


Pytorch-ResNet18
<img src="./perceptron/utils/images/doc/console_gaussianblur_minivan2ambulance.jpeg" style="zoom:50%;" />

<img src="./perceptron/utils/images/doc/pic_gaussianblur_minivan2ambulance.jpeg" style="zoom:50%;" />

Object detection:

PaddleHub-SSD
<img src="./perceptron/utils/images/doc/console_paddlehub_miss_gaussian_blur_target_class.jpg" style="zoom:50%;" />

<img src="./perceptron/utils/images/doc/pic_miss_gaussian_blur_target_class.jpg" style="zoom:80%;" />

PytorcHub-YOLOv5s
<img src="./perceptron/utils/images/doc/console_pytorchhub_miss_gaussian_blur_target_class.jpg" style="zoom:50%;" />

<img src="./perceptron/utils/images/doc/pic_pytorchhub_miss_gaussian_blur_target_class.jpg" style="zoom:1000%;" />


### 2.2 Script file call

Examples of script files for paddle and pytorch are given in the directory paddleshield/Robustness/examples.

- **Examples**

```shell
cd paddleshield/Robustness

# The results can be found under Robustness/examples/images.
python examples/paddle_sp_example.py
```

### 2.3 Batch attack

In order to facilitate users to verify multi-image attacks, this project provides code examples and outputs the results to a csv file. Users can draw relevant statistical graphs based on the output data.

- **Examples**：Imagenet dev dataset.

```shell
cd Robustness/batch_attack

# Download the imagenet dev dataset and store it under Robustness/batch_attack/ILSVRC2012_img_val

# The image labels are under Robustness/batch_attack/caffe_ilsvrc12.

python Batch_Launcher.py  --framework paddle --model resnet50 --criteria misclassification --metric gaussian_blur
```

- **Results**

<img src="./perceptron/utils/images/doc/batchattack_result_csv.jpeg" style="zoom:50%;" />

### 2.3 Evaluation

#### Image classification
We evaluate the robustness of image classification models on several **animal** images.

- **Results**

<img src="./perceptron/utils/images/doc/image_classification_robustness.jpg" style="zoom:70%;" />

#### Object detection
We evaluate the robustness of object detection models on several **pedestrian** images.

- **Results**

<img src="./perceptron/utils/images/doc/object_detection_robustness.jpg" style="zoom:70%;" />


## 3. User-defined model

To ease the trouble evaluating a customized deep learning models, we provide guidelines, and examples to port user's models and make them work with our perceptron robustness benchmark.

### 3.1 Cifar10 image classification model

Before evaluating the relevant classification model, make sure that the model implementation and weights are prepared. Here is based on resnet50, training a 10-class image classification model to give examples, and save the weights.

```shell
python Robustness/examples/User_Model/cifar_model.py
```

### 3.2 Adapt to user-defined models

We require users to create a subclass of `PaddleModel` or `PyTorchModel` and complete the `load_model()` method so that the model can be successfully loaded and evaluated. Here, we take the paddle model as an example to provide users with reference to the model trained in 3.1.

```python
from __future__ import absolute_import

import os
import paddle
from perceptron.models.classification.paddle import PaddleModel


class PaModelUpload(PaddleModel):
    def __init__(self,
                 bounds,
                 num_classes,
                 channel_axis=1,
                 preprocessing=(0, 1)):
        # load model
        model = self.load_model()
        model.eval()

        super(PaModelUpload, self).__init__(model=model,
                                            bounds=bounds,
                                            num_classes=num_classes,
                                            channel_axis=channel_axis,
                                            preprocessing=preprocessing)

    @staticmethod
    def load_model():
        """To be implemented...
           model evaluation participants need to implement this and make sure a paddle model can be loaded and fully-functional"""
        network = paddle.vision.models.resnet50(num_classes=10)
        model = paddle.Model(network)
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, '../../../examples/User_Model/checkpoint/test')
        print(model_path)
        model.load(model_path)
        model.network.eval()
        return model.network
```

### 3.3 Start evaluation

We take Brightness as an example of attack and provide evaluation code for user-defined models.

- **Examples**

```shell
cd paddleshield/Robustness

python examples/paddle_userupload_br.py
```

- **Results**

<img src="./perceptron/utils/images/doc/console_brightness_truck2bird.jpeg" style="zoom:60%;" />

<img src="./perceptron/utils/images/doc/pic_brightness_truck2bird.jpeg" style="zoom:60%;" />
