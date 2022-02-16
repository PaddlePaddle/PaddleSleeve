# Example for using GAN to reconstruct AT&T face training data
English | [简体中文](./README_cn.md)

As shown below, this example simulates how malicious participants recover training data from the other party through the GAN attack module in a  federated training scenario that sharing model parameters.

<p align="center">
  <img src="../../../docs/images/gan_example.png?raw=true" width="700" title="GAN attack in federated learning">
</p>


## Run Example

Install the `privbox` tool ([Installation](../../../README.md###Installation)) firstly, then run the example as:
```shell

python3 gan_inversion_with_at_face.py

```

The example provides the following parameters that the user can customize the settings.

- `--batch_size` (int, default=32): batch size of training data.
- `--attack_epoch` (int, default=100): epoch of GAN attack.
- `--target_label` (int, default=1): target label for reconstruction.
- `--learning_rate_real` (float, default=0.0002): learning rate for actual federated training.
- `--learning_rate_fake` (float, default=0.0002): learning rate for training model with fake data by attacker.
- `--learning_rate_gen` (float, default=0.0002): learning rate for training GAN's generator by attacker.
- `--result_dir` (str, default='./att_results'): results dir.
- `--num_pic_save` (int, default=4): number of pictures saving per epoch.


By default, the results are saved in the 'att_results/' directory where the running directory is located. User can also specify the location by `--result_dir`.


**Result Example 1**：

Input Parameters：

```shell
attack_epoch=100, batch_size=32, learning_rate_fake=0.0002, learning_rate_gen=0.0002, learning_rate_real=0.0002, num_pic_save=4, result_dir='./att_results', target_label=1
```

Attack Result：

Images for target (left picture) and reconstructed (right picture) are list as follows.

<p align="center">
  <img src="../../../docs/images/gan_target.png?raw=true" width="50" title="GAN target image"/>           
  <img src="../../../docs/images/gan_reconstruct.png?raw=true" width="50" height="60" title="GAN attack reconstructed image"/>
</p>