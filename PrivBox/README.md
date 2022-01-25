# PrivBox--Privacy Analysis Tools
English | [简体中文](./README_cn.md)


PrivBox is a Python library for testing AI model privacy leaking risk, which is based on PaddlePaddle. PrivBox provides multiple AI privacy attacks implementations of recent researches, including model inversion, membership inference, property inference and model extraction, aiming to help developer to find out privacy issues of AI models.

<p align="center">
  <img src="docs/images/PrivBox.png?raw=true" width="500" title="PrivBox Framework">
</p>


**PrivBox Supported Attack Methods**

<table>
   <tr>
      <td>Attacks</td>
      <td>Implemented Methods</td>
      <td>References</td>
   </tr>
   <tr>
      <td rowspan="2">Model Inversion</td>
      <td>Deep Leakage from Gradients（DLG)</td>
      <td><a href="https://arxiv.org/pdf/1906.08935.pdf">[ZLH19]</a></td>
   </tr>
   <tr>
      <td>Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning (GAN)</td>
      <td><a href="https://arxiv.org/pdf/1702.07464.pdf">[HAPC17]</a></td>
   </tr>
   <tr>
      <td rowspan="2">Membership Inference</td>
      <td>Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting</td>
      <td><a href="https://arxiv.org/pdf/1709.01604.pdf">[YGFJ18]</a></td>
   </tr>
   <tr>
      <td>ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models</td>
      <td><a href="https://arxiv.org/pdf/1806.01246.pdf">[SZHB19]</a></td>
   </tr>
   <tr>
      <td rowspan="1">Property Inference</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td rowspan="2">Model Extraction</td>
      <td>Knockoff Nets: Stealing Functionality of Black-Box Models</td>
      <td><a href=http://openaccess.thecvf.com/content_CVPR_2019/papers/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.pdf>[OSF19]</td>
   </tr>
</table>

## Getting Started


### Requirements
python >= 3.6

PaddlePaddle >= 2.0（[paddle installation](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)）

### Installation
In directory `PrivBox/`, run command

```
python3 setup.py bdist bdist_wheel
```

then the whl package is generated in `dist/` directory，execute

```
python3 -m pip install dist/privbox-x.x.x-py3-none-any.whl
```

to complete the installation. You can run examples in `examples` directory after installation.

### Examples

Directory [`examples/`](examples/) list multiple examples for usage PrivBox.


## Contributions

[Development Guide](docs/README.md)


PrivBox is under continuous development. Contributions, bug reports and other feedbacks are very welcome!

## References

\[ZLH19\] Ligeng Zhu, Zhijian Liu, and Song Han. Deep leakage from gradients. NeurIPS, 2019.

\[HAPC17\] Briland Hitaj, Giuseppe Ateniese, and Fernando P´erez-Cruz. Deep models under the gan: Information leakage from collaborative deep learning. CCS, 2017.

\[YGFJ18\] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. Computer Security Foundations Symposium (CSF), 2018.

\[SZHB19\] Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, Michael Backes. ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models. Network and Distributed Systems Security Symposium (NDSS) 2019.

\[OSF19\]Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz. Knockoff nets: Stealing functionality of black-box models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019