## Adversarial example detection

   In `PaddleSleeve/AdvBox/AEs_detection`, we implements the adversarial example detection function for image classification models. By using the local intrinsic dimensionality (LID) metric as the base algorithm, two optimized adversarial example detection algorithms based on model output and mixture gaussian are implemented on the basis of the LID algorithm. The LID metric is used to characterize the dimensionality of the adversarial subspace to reveal the essential difference between ordinary samples and adversarial examples. The model output-based approach uses model vector features, i.e., confidence output, to compute the regularized vector features. The mixture gaussian-based approach uses mixture gaussian loss to optimize the baseline model, and the optimized classifier is used to discriminate between normal and adversarial examples.


### Setup_paths

   Open `setup_paths.py` and set the paths and other detector-related settings.

### Run Adversarial example detection

   1. Traditional local intrinsic dimensionality(lid) adversarial sample detection method:
      Step-1 Train baseline classification, here we use the resnet34 model for cifar10 dataset:
         ```python
         python model_retrain.py
         ```
      Step-2 Generate the adversarial examples corresponding to the baseline model and save them in npy format:
         ```python
         python generate_adv.py
         ```
      Step-3 Run the adversarial example detection algorithm:
         ```python
         python detect_lid_paddle.py -d=cifar -a=pgdi_0.0625
         ```
   2. LID adversarial example detection method + baseline model output;
      Step-1 The first 2 steps are the same as the classical lid adversarial sample detection method
      Step-2 Run the adversarial example detection algorithm:
         ```python
         python detect_lid_paddle_logits.py -d=cifar -a=pgdi_0.0625
         ```
   3. LID adversarial example detection method + lgm mixture gaussian based baseline model optimization
      Step-1 Fine-tuning the baseline model with lgm loss
         ```python
         python model_retrain_lgm.py -d=cifar -a=pgdi_0.0625
         ```
      Step-2 The 2rd steps are the same as the classical lid adversarial sample detection method
      Step-3 Run the adversarial example detection algorithm:
         ```
         python detect_lid_paddle_lgm.py -d=cifar -a=pgdi_0.0625
         ```

Results
   Evaluation Index:
      1) Accuracy: The overall classification accuracy of the whole original and adversarial examples.
      2) FPR: False positive rate. It is measured by calculating the number of clean inputs that are detected as adversarial samples divided by the total number.
      of clean inputs. The lower the better.
      3) Pretrained accuray: The classification accuracy of adversarial examples on the baseline classifier.
      4) SAEs: The detection accuracy of the adversarial samples that are predicted to be positive and the actual positive samples (TPR), in which the adversarial samples are not recognized by the original classification model;
      5) FAEs: The detection accuracy of the adversarial samples that are predicted to be positive and the actual positive samples (TPR),in which the adversarial samples are not recognized by the original classification model.

 <table align="center">
 <tr>
    <td align="center">Accuracy</td>
    <td align="center">FPR</td>
    <td align="center">Pretrained accuray</td>
    <td align="center">SAEs </td>
    <td align="center">FAEs </td>
</tr>

<tr>
    <td align="center">LID</td>
    <td align="center">59.38%</td>
    <td align="center">27.47%</td>
    <td align="center">12.15%</td>
    <td align="center">47.42%</td>
    <td align="center">37.67%</td>
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
    <td align="center">83.23%</td>
    <td align="center">6.78%</td>
    <td align="center">12.15%</td>
    <td align="center">77.64%</td>
    <td align="center">41.44%</td>
</tr>

<tr>
    <td align="center">LID_lgm_logit</td>
    <td align="center">85.10%</td>
    <td align="center">6.78%</td>
    <td align="center">19.23%</td>
    <td align="center">75.84%</td>
    <td align="center">62.34%</td>
</tr>

</table>
