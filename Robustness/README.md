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

# keras: The results can be found under Robustness/images.
python perceptron/launcher.py --framework keras --model ssd300 --criteria target_class_miss --metric gaussian_blur --image example.jpg --target_class -1

python perceptron/launcher.py  --framework keras --model resnet50 --criteria misclassification --metric gaussian_blur --image example.jpg

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

## Data Augmentation

As a common trick to enhance model robustness, data augmentation, which transforms original input image into a set of slightly perturbed variants, is widely-used in ML tasks. In Robustness, we also provide a data augmentation module. It is implemented with a variety of base augmentation operators, covering 8 categories, over 40 types of common data augmentation methods, easy and ready to use. Users can also compose basic operators for more complex augmentations. For each operators, three levels of augmenting magnitude are available. 

This module is intended for enhancing the robustness of DNN models. It can be easily integreted into any DNN models to improve the performance, especially in the case of limited data. Note that although the demo uses models on PaddlePaddle platform, this module is also compatible with other mainstream ML platforms, including but not limited to TensorFlow and PyTorch, etc. 

One particular scenario in which data augmentation is needed is autonomous driving. Given the real road condition is complicated and constantly-changing, the object detection model in autonoums driving vehicles must be able to correctly detect all objects on the road under any circumstances. That is, the model must give correct results even on distorted input images. In the following section, a list of images showing a car plate after all types of augmentations are exhibited. Hopefully this can give a better sense of what the effect of each augmentation looks like, and in what scenario image augmentations may be applied.

## 1. Basic Operators Inventory

Here, we showcased all 42 types of aumentations, divided into 8 categories: Deformation, Geometry Transformation, Pattern, Blur, Additive Noise, Weather, Image Processing, and Compression & Smoothing. The implementation of Curve, Stretch, Rectangle Grid, and some other augmentations are based on **Straug** [https://github.com/roatienza/straug]. Implementations of GridDistortion, HueSaturation, and some other augmentations are based on **Albumentations** [https://github.com/albumentations-team/albumentations].

<table>
  <tr><td align="center">Original image</td></tr>
  <tr><td align="center"><img src="./perceptron/augmentations/images/car_plate.jpg" width=200></td></tr>

</table>

- **Deformation**
<table>
  <tr>
    <td align="center">Curve</td>
    <td align="center">Distort</td>
    <td align="center">Stretch</td>
    <td align="center">GridDistortion</td>
    <td align="center">OpticalDistortion</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Curve.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Distort.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Stretch.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/GridDistortion.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/OpticalDistortion.jpg" width=150></td>
</tr>

</table>

- **Geometry Transformation**
<table>
  <tr>
    <td align="center">Rotate</td>
    <td align="center">Perspective</td>
    <td align="center">Transpose</td>
    <td align="center">Translation</td>
    <td align="center">RandomCrop</td>
    <td align="center">RandomMask</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Rotate.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Perspective.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Transpose.jpg" height=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Translation.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/RandomCrop.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/RandomMask.jpg" width=150></td>
</tr>
</table>

- **Pattern**
<table>
  <tr>
    <td align="center">VerticalGrid</td>
    <td align="center">HorizontalGrid</td>
    <td align="center">Rectangle/EllipticalGrid</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/VGrid.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/HGrid.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/RectGrid.jpg" width=150></td>
</tr>
</table>

- **Blur**
<table>
  <tr>
    <td align="center">GaussianBlur</td>
    <td align="center">MedianBlur</td>
    <td align="center">DefocusBlur</td>
    <td align="center">GlassBlur</td>
    <td align="center">ZoomBlur</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/GaussianBlur.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/MedianBlur.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/DefocusBlur.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/GlassBlur.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/ZoomBlur.jpg" width=150></td>
</tr>
  
<tr>
     <td align="center">MotionBlur</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/MotionBlur.jpg" width=150></td>
</tr>
</table>

- **Additive Noise**
<table>
  <tr>
    <td align="center">GaussianNoise</td>
    <td align="center">ShotNoise</td>
    <td align="center">ImpulseNoise</td>
    <td align="center">SpeckleNoise</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/GaussianNoise.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/ShotNoise.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/ImpulseNoise.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/SpeckleNoise.jpg" width=150></td>
</table>

- **Weather**
<table>
  <tr>
    <td align="center">Fog</td>
    <td align="center">Rain</td>
    <td align="center">Snow</td>
    <td align="center">Shadow</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Fog.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Rain.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Snow.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Shadow.jpg" width=150></td>
</tr>
</table>

- **Image Processing**
<table>
  <tr>
    <td align="center">Contrast</td>
    <td align="center">Brightness</td>
    <td align="center">Sharpness</td>
    <td align="center">Posterize</td>
    <td align="center">Solarize</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Contrast.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Brightness.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Sharpness.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Posterize.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Solarize.jpg" width=150></td>
</tr>
 
<tr>
    <td align="center">Color</td>
    <td align="center">HueSaturation</td>
    <td align="center">Equalize</td>
    <td align="center">Invert</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Color.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/HueSaturation.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Equalize.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Invert.jpg" width=150></td>
 </tr>
</table>

- **Smoothing & Compression**
<table>
  <tr>
    <td align="center">JPEGCompression</td>
    <td align="center">Pixelate</td>
    <td align="center">BitReduction</td>
    <td align="center">MaxSmoothing</td>
    <td align="center">AverageSmoothing</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/JPEG_Compression.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/Pixelate.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/BitReduction.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/MaxSmoothing.jpg" width=150></td>
    <td align="center"><img src="./perceptron/augmentations/output/showcase/AvgSmoothing.jpg" width=150></td>
</tr>
</table>

## 2. Usage

Users should use instances of SerialAugment class to compose base operators or to apply them. These operators can function on either tensors or images. In the former case, the input should be a numpy array, or it can be a path pointing to an image or an image folder in the file system, as in the second case. Detailed explaination with demos can be found below. 

### 2.1 Initialization
- ### Create a SerialAugment Instance

  The SerialAugment class is the interface between users and the aumentation operators. 

  `class SerialAugment(transforms=[], format='CHW', bound=(0, 1), input_path=None, output_path=None)`

 - **Parameters**
     - `transforms (Dict|list(Dict))`  
     : The types of augmentation to use. This should be a list of Dicts, and each Dict specifies the class name and initialization arguments for one operator
     - `format (str)`  
     : The data format of the inputs. Should be 'HWC' or 'CHW'. Default: 'CHW'. 
     - `bound (tuple)`
     : A tuple of integers that specifies the range of input data. Default: (0, 1)
     - `input_path`
     : This specifies the path to input images. Default: None
     - `output_path`
     : The place to save the augmented images. Default: None
     
 - **Examples**
 
 The following code segment instantiates an augment operator that draws elliptical grids on the input image.
 ```python
     # Import the interface class
     from perceptron.augmentations.augment import SerialAugment
     operator = [{'RectGrid': {'ellipse': True}}]
     data_augment = SerialAugment(operator, format='HWC', bound=(0, 255))

 ```
 
- ### Compose Operators

In addition to the pre-implemented operators, users can also combine them to achieve more complex augmentation effect. To compose base operators, simply pass to the SerialAugment constructor the list of base augmentations you want to use. All augmentations declared will be applied in sequence to the same image. 

 - **Examples**
 
 The following code composes Rotate and GaussianNoise. When applied to an image, it will rotate the image then add Gaussian noise to it.
```python
     # Import the interface class
     from perceptron.augmentations.augment import SerialAugment
     operator = [{'Rotate': {}},
                 {'GaussianNoise': {}}]
     data_augment = SerialAugment(operator, format='HWC', bound=(0, 255))

 ```

### 2.2 Quick Trial: Augmentations with I/O

After initializing the data augmentor, users can directly apply it to some images. There are two ways to specify the images to be augmented: users can set the input path by either passing it to the constructor, or by calling `set_image(path)` funtion. The SerialAugment instances are callable objects, so once the input images are set, augmentation can be initiated by calling the object. 

`def __call__(img=None, mag=0)`

**Parameters**
  - `img (numpy.array|list)`: Images to be augmented. if None, then the augmentor will load images specified by `set_image` method from file system.
  - `mag (int)`: The augmentation magnitude. Must be one of 0, 1, or 2. Level 0 for the modest augmentation and level 2 for the most radical augmentation.

This function returns augmented images. It can also save augmented images to the place specified by users. Same as input path, the output path can be set either by the constructor, or via `set_out_path(path)` function.

**Examples**

The above example continues. Following code snippet applies the composed Rotate & GaussianNoise augmentor to images in `images/demo` folder, and save augmented images to the `output/out_demo` folder.

```python
  operator = [{'Rotate': {}},
              {'GaussianNoise': {}}]
  data_augment = SerialAugment(operator, format='HWC', bound=(0, 255))
  
  # set the images to be augmented 
  data_augment.set_image('images/demo')
  
  # set the output path
  data_augment.set_out_path('output/out_demo')
  
  # augmentation starts
  augmented_images = data_augment(mag=0)
```

If successful, there will be a new folder `output_demo`, in which all augmented images are saved. 

<img src="./perceptron/augmentations/images/doc/IO_1.png" width=300 height=200 />      <img src="./perceptron/augmentations/images/doc/IO_2.png" width=300 height=200 />

Visualize the augmentation effect 
<table>
  <tr>
    <td align="center">Original</td>
    <td align="center">Augmented</td>
</tr>
  
<tr>
    <td align="center"><img src="./perceptron/augmentations/images/doc/van.jpg" width=500></td>
    <td align="center"><img src="./perceptron/augmentations/images/doc/augmented_van.jpg" width=500></td>
</tr>
</table>

### 2.3 Incorporated in Model Training

This module is intended for enhancing the performance of DNN models. It can be smoothly incorporated in model training as part of the data preprocessing. Users can either augment each minibatch of data manually, which provides more flexibility in case only part of training data should be augmented. Or users may choose to incorporate the data augmentor in the dataloader, easier to implement and faster in speed. 

 - ### Augment Single MiniBatch

During training, users can augment a single mini-batch of data by calling the augmentor and passing in the data as argument. Suppose `data_augment` is a SerialAugment instance, and `ori_data` is a mini batch of data, then run the following command to augment the data. 
```python
  augmented_data = data_augment(ori_data, mag=0)
```
- **demo**

  The following code manually augments each mini-batch of training data. This demo trains PaddlePaddle's ResNet34 model on Cifar10 dataset.
  
  ```python
  
    ...
 
    # Initialize dataset, loader, model, optimizer, etc. 
    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=T.Transpose(order=(2, 0, 1)), backend='cv2')

    model = paddle.vision.models.resnet34(pretrained=False, num_classes=10)
    BATCH_SIZE = 128
    train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    learning_rate = 0.001
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()
    normalize_fn = lambda x_batch: [T.normalize(x, std=[62.99, 62.08, 66.7], mean=[125.31, 122.95, 113.86]) for x in x_batch]

    # Initialize data augmentor
    data_augment = SerialAugment(transforms=[{'Rotate': {}},
                                             {'Translation': {}},
                                             {'RandomCrop': {}}],
                                 format='CHW', bound=(0, 255))

    num_epoch = 20
    model.train()

    # Start training
    for epoch in range(num_epoch):
        for i, data in enumerate(train_loader):
            img, label = data

            # Start augmenting data
            aug_img = paddle.unstack(img)
            aug_img = data_augment(aug_img, mag=0)
            aug_img = normalize_fn(aug_img)
            aug_img = paddle.to_tensor(np.stack(aug_img))

            # Get inference result
            pred = model(aug_img)
            # Backward propagate loss and update model parameters
            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            
   ...
   
   ```

- ### Combine with Dataloader

Operators in the data augmentation module has a special optimization for PaddlePaddle framework that allows them to be deployed with only one line of code. By simply adding an instance of SerialAugment to dataloaders' list of `transform` operators, data augmentation will be incorporated into dataloaders' preprocessing procedures frictionlessly. 

- **demo**

The following example demonstrates the entire process of using augmented images to train PaddlePaddle's ResNet34 model on Cifar10 dataset. 

```python

def train():
    data_augment = SerialAugment(transforms=[{'Rotate': {}},
                                             {'Translation': {}},
                                             {'HueSaturation': {}},
                                             {'GridDistortion': {}}],
                                 format='HWC', bound=(0, 255))
    
    # Prepending data_augment module to dataloaders' list of transform operators
    train_transform = T.Compose([data_augment,
                                T.Normalize(mean=[125.31, 122.95, 113.86], std=[62.99, 62.08, 66.7], data_format='HWC'),
                                T.Transpose(order=(2, 0, 1))])
    
    test_transform = T.Compose([T.Normalize(mean=[125.31, 122.95, 113.86], std=[62.99, 62.08, 66.7], data_format='HWC'),
                                T.RandomRotation(30),
                                T.HueTransform(0.2),
                                T.RandomCrop(size=32, padding=4),
                                T.Transpose(order=(2, 0, 1))])

    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=train_transform, backend='cv2')
    val_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=test_transform, backend='cv2')

    model = paddle.vision.models.resnet34(pretrained=False, num_classes=10)
    model = paddle.Model(model)
    BATCH_SIZE = 128
    train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = paddle.io.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    learning_rate = 0.001
    loss_fn = paddle.nn.CrossEntropyLoss()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    model.prepare(optimizer=opt, loss=loss_fn, metrics=paddle.metric.Accuracy())

    model.fit(train_loader, test_loader, batch_size=128, epochs=20, eval_freq=5, verbose=1)
    model.evaluate(test_loader, verbose=1)
```

The source code is available in `perceptron/augmentations/cifar10_dataaug_tutorial_dataloader.py`
