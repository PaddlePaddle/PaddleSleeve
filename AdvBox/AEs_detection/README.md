# Adversarial example detection

## 1. LID (local intrinsic dimensionality) related detection algorithms

  In `PaddleSleeve/AdvBox/AEs_detection/LID`, we implements the adversarial example detection function for image classification models, which contain two detetcion algorithms for global noise and patch noise adversarial examples. For global noise adversarial example detection, by using the local intrinsic dimensionality (LID) metric as the base algorithm, two optimized adversarial example detection algorithms based on model output and mixture gaussian are implemented on the basis of the LID algorithm. For patch noise adversarial example detection, based on LID metric, the feature norm in the feature map is significantly higher at the corresponding adversarial patch position than at other positions, and also significantly higher than the maximum feature norm of the normal sample. During the process of LID feature extraction, the high norms of the adversarial patch is used for feature enhancement, which increases the distinguishability between the adversarial sample and the normal sample. The LID metric extracted after feature enhancement with the high feature norm of the adversarial patch is used for adversarial example detection to improve the detection performance of the patched adversarial samples. The LID metric is used to characterize the dimensionality of the adversarial subspace to reveal the essential difference between ordinary samples and adversarial examples. The model output-based approach uses model vector features, i.e., confidence output, to compute the regularized vector features. The mixture gaussian-based approach uses mixture gaussian loss to optimize the baseline model, and the optimized classifier is used to discriminate between normal and adversarial examples.


### Setup_paths

   Open `setup_paths.py` and set the paths, attack methods and other detector-related settings.

### Run Adversarial example detection

#### Global noise adversarial example detection

   1. Traditional local intrinsic dimensionality(LID) adversarial sample detection method.    
      **Step-1** Train baseline classifier for cifar10 dataset, here we use the pretrained resnet34 model.     
         ```python
         python model_retrain.py
         ```
      **Step-2** Generate the adversarial examples corresponding to the baseline model and save them in npy format. Here we use pgd attack algorithm. The attack algorithms are cited from `PaddleSleeve/AdvBox/attacks`.
         ```python
         python generate_adv.py
         ```
      **Step-3** Run the adversarial example detection algorithm.  
         ```python
         python detect_lid.py -d=cifar -a=pgdi_0.0625
         ```
   2. LID adversarial example detection method + baseline model output.  
      **Step-1** The first 2 steps are the same as the classical lid adversarial sample detection method.  
      **Step-2** Run the adversarial example detection algorithm.    
         ```python
         python detect_lid_logits.py -d=cifar -a=pgdi_0.0625
         ```
   3. LID adversarial example detection method + lgm mixture gaussian based baseline model optimization.  
      **Step-1** Fine-tuning the baseline model with lgm loss.  
         ```python
         python model_retrain_lgm.py -d=cifar -a=pgdi_0.0625
         ```
      **Step-2** The 2rd steps are the same as the classical lid adversarial sample detection method.  
      **Step-3** Run the adversarial example detection algorithm.  
         ```
         python detect_lid_lgm.py -d=cifar -a=pgdi_0.0625
         ```

#### Patch noise adversarial example detection  
   **Step-1** Train baseline classifier for cifar10 dataset, here we use the pretrained resnet34 model.  
      
        python model_retrain.py      
   **Step-2** Generate the patch noise adversarial examples corresponding to the baseline model and save them in npy format. Here we use patch adversarial attack algorithm. The attack algorithms are cited from `PaddleSleeve/AdvBox/attacks`.     
      
        python generate_adv_patch.py   
   **Step-3** Run the adversarial example detection algorithm.   
   
        python detect_auglid.py -d=cifar -a=patch
      

### Results
   **Evaluation Index**:  
   
         1) Accuracy: The overall classification accuracy of the whole original and adversarial examples.  
         2) FPR: False positive rate. It is measured by calculating the number of clean inputs that are detected as adversarial samples divided by the total number of clean inputs. The lower the better.   
         3) Pretrained accuray: The classification accuracy of adversarial examples on the baseline classifier.  
         4) SAEs: The detection accuracy of the adversarial samples that are predicted to be positive and the actual positive samples (TPR), in which the adversarial samples are not recognized by the original classification model.  
         5) FAEs: The detection accuracy of the adversarial samples that are predicted to be positive and the actual positive samples (TPR),in which the adversarial samples are not recognized by the original classification model.  

#### Global noise adversarial example detection:

 <table align="center">
 <tr>
    <td align="center">Model</td>
    <td align="center">Accuracy</td>
    <td align="center">FPR</td>
    <td align="center">Pretrained accuray</td>
    <td align="center">SAEs </td>
    <td align="center">FAEs </td>
</tr>

<tr>
    <td align="center">LID</td>
    <td align="center">54.68%</td>
    <td align="center">6.7%</td>
    <td align="center">12.15%</td>
    <td align="center">16.30%</td>
    <td align="center">14.38%</td>
</tr>

<tr>
    <td align="center">LID_lgm</td>
    <td align="center">67.70%</td>
    <td align="center">27.47%</td>
    <td align="center">19.23%</td>
    <td align="center">44.67%</td>
    <td align="center">52.81%</td>
</tr>

<tr>
    <td align="center">LID_logit</td>
    <td align="center">82.71%</td>
    <td align="center">2.08%</td>
    <td align="center">12.15%</td>
    <td align="center">72.19%</td>
    <td align="center">33.56%</td>
</tr>

<tr>
    <td align="center"><b>LID_lgm_logit</td>
    <td align="center"><b>85.10%</td>
    <td align="center"><b>6.78%</td>
    <td align="center"><b>19.23%</td>
    <td align="center"><b>75.84%</td>
    <td align="center"><b>62.34%</td>
</tr>

</table>  


#### Patch noise adversarial example detection:

 <table align="center">
 <tr>
    <td align="center">Model</td>
    <td align="center">Accuracy</td>
    <td align="center">FPR</td>
    <td align="center">Pretrained accuray</td>
    <td align="center">SAEs </td>
    <td align="center">FAEs </td>
</tr>

<tr>
    <td align="center">augLID</td>
    <td align="center">64.04%</td>
    <td align="center">11.07%</td>
    <td align="center">63.55%</td>
    <td align="center">40.18%</td>
    <td align="center">38.57%</td>
</tr>

<tr>
    <td align="center"><b>augLID_logit</td>
    <td align="center"><b>71.62%</td>
    <td align="center"><b>12.85%</td>
    <td align="center"><b>63.55%</td>
    <td align="center"><b>62.33%</td>
    <td align="center"><b>52.52%</td>
</tr>


</table>


### Citation  

    @article{ma2018characterizing,  
    title={Characterizing adversarial subspaces using local intrinsic dimensionality},  
    author={Ma, Xingjun and Li, Bo and Wang, Yisen and Erfani, Sarah M and Wijewickrema, Sudanthi and Schoenebeck, Grant and Song, Dawn and Houle, Michael E and Bailey, James},  
    journal={arXiv preprint arXiv:1801.02613},  
    year={2018}  
    }    
    @inproceedings{LGM2018,
    title={Rethinking Feature Distribution for Loss Functions in Image Classification},
    author={Wan, Weitao and Zhong, Yuanyi and Li, Tianpeng and Chen, Jiansheng},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
    }
    @inproceedings{yu2021defending,
    title={Defending against universal adversarial patches by clipping feature norms},
    author={Yu, Cheng and Chen, Jiansheng and Xue, Youze and Liu, Yuyang and Wan, Weitao and Bao, Jiayu and Ma, Huimin},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={16434--16442},
    year={2021}
    }
