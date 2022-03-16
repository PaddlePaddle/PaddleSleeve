# Advbox - Object Detection 
Advbox-ObjectDetection is a sub-module of Advbox, which carries out adversarial attacks & defenses on object detection models. It provides users numerous attacking methods, both blackbox and whitebox, that can be deployed to attack object detection models as well as popular defense methods against intended adversaries to protect these models. Additionally, the models repository in this sub-module includes pretrained version of most of the mainstream object detection models that can be used for users' own attacking or defensing test.

This sub-module is designed primarily for object detection models on PaddleDection platform. It depends on the configs directory from PaddleDetection for model configurations and ppdet module from paddledet package for model structure definitions and some auxiliary utility tools. Note that paddledet-2.3.0 is hard-copied into this project, so users need not to install paddledet themselves. Future developers are also welcomed to add their own models on other platforms if desired. 

For object detection tasks, adversarial attackes roughly falls into two categroies: perturbation on the entire image and patch attack. Perturbation on the entire iamge tries to generate adversarial examples by adding a small perturbation to the entire image. Adversaries are allowed to change every pixels in the image, but each pixes can only vaires by a small size. The overall difference between original images and adversarial examples should be small enough to be negligible to human. On the other hand, patch attack aims at fooling the detection model by adding a patch to the input image. The patch can be add to any places inside the image, and can have arbitrary pattern, but its size is restricted. Unless specified otherwise, the term 'attack' refers to entire-image perturbation attack in this document. More details about attacks and patch attacks, including corresponding demos could be found below. 

## Attack on object detection models 
Adversaries in this subpart is allowed to perturb the entire image, but the amplitude of perturbation is confined. This module provides single_model attack, ensemble attack, and defensive adversarial training under such circumstance. Detailed description and usage can be found below.

### Attacks on single model
In Advbox-ObjectDetection, two well-known whitebox attacking methods, CW and PGD, are implemented, and a command-line tool is given to generate adversarial examples with Zero-Coding.

**Usage:** 
- The python file `launcher.py` in `attacks/single_attack` directory is provided to launch an adversarial attack. Users can specify the attacking method, target model, input images, and other parameters via command-line arguments. 

- **Command-line parameters**
  - `--model`：Select the pre-trained model. Currently, we support ppdet's pretrained yolov3, faster_rcnn, cascade_rcnn, and detr models with various backbones. 
  ```python
  "models": [
    # "paddledet_yolov3_darknet53",
    # "paddledet_yolov3_mobilenet_v1",
    # "paddledet_yolov3_mobilenet_v3_large",
    # "paddledet_yolov3_resnet50vd",
    # "paddledet_faster_rcnn_resnet50",
    # "paddledet_faster_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnet101_vd_fpn",
    # "paddledet_cascade_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnext101_64x4d_dcn",
    # "paddledet_detr_resnet50",
    # "paddledet_deformable_detr_resnet50"
  ]
  ``` 
  
  - `--image`：Users can upload their own pictures to the obj_detection/attacks/utils/images folder of the original pictures being attacked.
  - `--criteria`：Provides adversarial criterions, currently support `target_class_miss`.
  - `--target_class`: The category of the object in the input image.
  - `--metric`：Provides different attackingm method. Currently support `carlini_wagner` and `pgd`.
  - `--distance`: The distance metric to be used, choose from `l2` or `linf`

- **Examples**
  - The following example attacks `yolov3_darknet53` models using cw attack, and l2 distance is used. 
  ```shell
  # Navigate to the correct directory 
  cd obj_detection/attack/single_attack

  # paddle：The results can be found under obj_detection/attacks/examples/images.
  python attack/single_attack/launcher.py  --model paddledet_yolov3_darknet53 --criteria target_class_miss --target_class 3 --metric carlini_wagner --image motor.jpg --distance l2 
  ```
  The above attack outputs the following summary, and a comparasion between original image and the adversarial example is plotted for better visulization effect.
  
  <img src="attack/outputs/images/single_cw_stdout.png" style="zoom:70%;" />
  <img src="attack/outputs/images/single_cw_demo.png" style="zoom:70%;" />

      
### Attack Transferability Eval & Ensemble Attacks
Adversarial examples are particularly useful because of its transferability (i.e. the adv example trained on one model can also fool another model.) Advbox provides automatic scripts to evaluate the transferabilty of cw and pgd attack. These scripts can be found in `ensemble_attack` folder. 

- **Serial Ensemble Attack**
  
  Serial ensemble attack takes in as input a list of models, and tries to attack them in sequence. The successful adversarial example of the previous model is used as the input to the later model. The successful adversarial example of the last model is passed in the victim model for transferability test.
  
  - **Usage**
  
    This evaluation script can be executed by the following command
    ```shell
    cd obj_detection/attack/ensemble_attack
    python serial_attack_eval.py
    ```
    Note that this script does not take command-line arguments. Users should modify the model list<sup>*</sup> and attack settings specification in the code in order to fit their own task. `dataset_dir`is the path to the input images. The default folder contains about 30 images. Users can add/remove images from the folder, or switch to their own test dataset if desired. 
    ```python
  
    ...
  
    # Choose your model here
    # "paddledet_yolov3_darknet53",
    # "paddledet_yolov3_mobilenet_v1",
    # "paddledet_yolov3_mobilenet_v3_large",
    # "paddledet_yolov3_resnet50vd",
    # "paddledet_faster_rcnn_resnet50",
    # "paddledet_faster_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnet101_vd_fpn",
    # "paddledet_cascade_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnext101_64x4d_dcn",
    # "paddledet_detr_resnet50",
    # "paddledet_deformable_detr_resnet50"
    model_list = ["paddledet_yolov3_darknet53",
                  "paddledet_faster_rcnn_resnet50_fpn"]
    victim_model_name = "paddledet_yolov3_mobilenet_v3_large"
    
    # Change here if you want to use your own test dataset
    dataset_dir = os.path.dirname(os.path.realpath(__file__ + '../../')) + '/utils/images/ensemble_demo'

    # Change the attacking config below
    dist = 'l2'
    attack_method = 'carlini_wagner'
    eps = 10 / 255
    num_test = 20

    # You need not change anything below
    
    ...
    
    ```
  - **Examples**

   The code snippiet above tries to attack yolov3_darknet53 and faster_rcnn_resnet50 networks in seriers. The adversatial example is then passed into another object detection network yolov3_mobilentv3 to test the transferability of the adversarial example. 
   
   The sample output will be similar to the following. 
   
  <img src="attack/outputs/images/serial_stdout.png" style="zoom:70%;" />
  
  A comparasion plot is shown to demonstrate the transferability of the adversarial examples. It proves that the adversarial example trained on one model can also, at least to some extent, fool other models.
  
    <img src="attack/outputs/images/serial_example.png" style="zoom:70%;" />


   
- **Weighted Ensemble Attack PGD**

  Another method of ensemble attack is to attack all models at the same time. The prediction results from all models are take into account when stepping the adversarial example. It is also possible to assign different weight to to models to be attacked. Models with larger weight may affect the adversarial example more greatly. 
  
  - **Usage**

    Use the following command to run the PGD ensemble attack
    ```shell
    python weighted_ensemble_attack_pgd.py
    ```
    Similar as in serial ensemble attack, users are welcome to change the list of models to be attacked<sup>*</sup> as well as other configs directly in the code. 
    ```python
    # choose from available models:
    # "paddledet_yolov3_darknet53",
    # "paddledet_yolov3_mobilenet_v1",
    # "paddledet_yolov3_mobilenet_v3_large",
    # "paddledet_yolov3_resnet50vd",
    # "paddledet_faster_rcnn_resnet50",
    # "paddledet_faster_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnet101_vd_fpn",
    # "paddledet_cascade_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnext101_64x4d_dcn",
    # "paddledet_detr_resnet50",
    # "paddledet_deformable_detr_resnet50"
    model_list = ['paddledet_yolov3_darknet53',
                  'paddledet_detr_resnet50',
                  'paddledet_faster_rcnn_resnet50_fpn']
    weight_list = [1, 1, 1]

    victim_model_name = 'paddledet_yolov3_mobilenet_v3_large'
    
    ... 
    
    ```
    
  - **Examples**

    The above code snippet demonstrates an example that ensembles attack on yolov3_darknet, faster_rcnn, and detr models to attack yolov3_mobilenet. 
    
  <img src="attack/outputs/images/weighted_pgd_stdout.png" style="zoom:70%;" />
  
  Here is the adversarial example with best transerability 
  
  <img src="attack/outputs/images/weighted_pgd_example.png" style="zoom:70%;" />

- **Weighted Ensemble Attack CW**

  This script launches cw ensemble attack. Its usage is basically the same as weighted_ensemble_attack_pgd
  
  - **Examples**

    ```python 
    # choose from available models:
    # "paddledet_yolov3_darknet53",
    # "paddledet_yolov3_mobilenet_v1",
    # "paddledet_yolov3_mobilenet_v3_large",
    # "paddledet_yolov3_resnet50vd",
    # "paddledet_faster_rcnn_resnet50",
    # "paddledet_faster_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnet101_vd_fpn",
    # "paddledet_cascade_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnext101_64x4d_dcn",
    # "paddledet_detr_resnet50",
    # "paddledet_deformable_detr_resnet50"
    model_list = ['paddledet_yolov3_darknet53',
                  'paddledet_detr_resnet50',
                  'paddledet_faster_rcnn_resnet50_fpn']
    weight_list = [1, 1, 1]

    victim_model_name = 'paddledet_yolov3_mobilenet_v3_large'
    ```
    The outputs are as follows. 
    
  <img src="attack/outputs/images/weighted_cw_stdout.png" style="zoom:70%;" />
    
  <sup>*</sup>*Note:* The order to initiate the list of models matters in some case. Since the config files are shared globally, if two models have the same structure and the one with non-default settings is initiated first, these settings will shadow the default settings used by the other model. This can result in model structure and parameters mismatch. In general, try to initiate the model with simpler structure first. 
  
     *e.g.* Declear your model list as `[faster_rcnn_resnet50, faster_rcnn_resnet50_fpn]` instead of `[faster_rcnn_resnet50_fpn, faster_rcnn_resnet50]`

### Defense
As malice adversaries increasingly threat AI security, people have proposed numerous methods to defense the deep learning netweork against the attack. Advbox also provides such defensive tools. The codes for defensive adversarial training are in `obj_detection/attack/defense` directory, and a script `advtrain_launcher.py` that allows user to launch an adversarial training from command-line arguments can also be found there. 

- **Usage**

  - **Config file**
  
    The launcher uses yaml files for model declaration and training configurations. These files can be found in `obj_detection/configs` directory, and users can add their own model there. If users want to construct their models on the base of other models, simply declear the base config file at the beginning of their own file and redefine the parameters they would like to overwrite. 
    
    A sample config file `yolov3_darknet53_advtrain`
      ```yaml
      _BASE_: [
      './yolov3_darknet53_270e_coco.yml',
      ]

      epoch: 540

      LearningRate:
        base_lr: 0.0005
        schedulers:
         - !PiecewiseDecay
           gamma: 0.1
           milestones:
            - 432
            - 486
        - !LinearWarmup
          start_factor: 0.
          steps: 8000

      TrainReader:
        batch_size: 4
        collate_batch: False

      TrainDataset:
        !COCODataSet
          image_dir: train2017
          anno_path: annotations/instances_val2017.json
          dataset_dir: dataset/coco
          data_fields: ['image', 'gt_bbox', 'gt_class', 'gt_score', 'is_crowd']

      EvalDataset:
        !COCODataSet
          image_dir: val2017
          anno_path: annotations/instances_val2017.json
          dataset_dir: dataset/coco

      ``` 

  - **Command-line parameters**
    - `--model`：Select the model to be defended. This should be the name of your yaml config file.
    - `--defense`: Selecte the defense method, support `natural_advtrain` and `free_advtrain`.
    - `--pretrained`: Whether to use pretrained models. If true, ppdet's pretrained weight for the corresponding model will be loaded
    - `--pretrained_weight`: The path to users' own pretrained weight. This only works when `--pretrained` flag is set to False
  
  - **Examples**
  
    Users can initiate an adversarial training by the following command
    ```shell
    cd obj_detection/attack/defense
    python advtrain_launcher.py --model yolov3_darknet53_advtrain --defense natural_advtrain --pretrained True
    ```
    The script will save the model after each epoch. Both the model and optimizer are saved in `obj_detection/attack/outputs/models/` directory



## Patch attack on object detection models 

In `PaddleSleeve/AdvBox/obj_detection/patch_attack`, we achieve the extended EOT attacking algorithm, a method using EOT to produce perturbation
to make the detection method give the wrong detection results. Specifically, we optimize the loss function
definition using the similar top three labels and top three logits to decrease the object confidence, and 
thus, the attack is able to succeed.


- A kindly Reminder: since paddlepaddle <= 2.1 does not support gradient backward for
 `paddle.nn.SyncBatchNorm` in eval() mode, to run the demonstration, we need to modify 
 all `sync-bn` components in detector model into `bn` (because `paddle.nn.BatchNorm` 
 supports gradient backward in eval() mode).
 
 If you want to customize your own demo script, you should try the following methods:
 
- For object detector like `configs/yolov3/_base_/yolov3_darknet53.yml`,
 add `norm_type: bn` on the third line.
- For object detector like `configs/ppyolo/ppyolo_mbv3_large_coco.yml`, add `norm_type: bn` 
 on the 9 th line.

### Run Target Patch Demonstration
After changing all `sync-bn` components into `bn`, run the following commandlines.

1. `cd PaddleSleeve/AdvBox/obj_detection/patch_attack`
adversial example train:
single attack: respectively use ppyolo, yolov3 and ssd detection models:

2. `python target_patch_eto_ppyolo.py -c ../configs/ppyolo/ppyolo_mbv3_large_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams --infer_img=dataloader/car_05.jpeg`
 
3. `python target_patch_eto_yolov3.py -c ../configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams --infer_img=dataloader/car_05.jpeg`

4. `python target_patch_eto_ssd.py -c ../configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml -o weights=https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams --infer_img=dataloader/car_05.jpeg`

ensemble attack: using ppyolo and yolov3 detection models:

5. `python target_patch_eto_ensemble.py -c ../configs/ppyolo/ppyolo_mbv3_large_coco.yml,../configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams,https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams --infer_img=dataloader/car_05.jpeg`
Note: the origin image, the added patch size and position and the object label shoud be manually given for specific images.


The successful execution of the `target_patch_eto_ppyolo.py`, will produce the following outputs.

**Image Compares**

<table align="center">
<tr>
    <td align="center"><img src="./patch_attack/dataloader/car_05.jpeg" width=300></td>
    <td align="center"><img src="./patch_attack/output/out_ppyolo_car_05.jpeg" width=300></td>
    <td align="center"><img src="./patch_attack/output/ppyolo_adverse_car_05.jpeg" width=300></td>
    <td align="center"><img src="./patch_attack/output/out_ppyolo_adverse_car_05.jpeg" width=300></td>
</tr>

<tr>
    <td align="center">Original Image</td>
    <td align="center">Original Detection Results</td>
    <td align="center">Acquired Adversarial Image </td>
    <td align="center">Adversarial Detection Results </td>
</tr>
</table>





