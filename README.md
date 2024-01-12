# IMGS
This repository contains the PyTorch code for the paper:

**Image Mixing and Gradient Smoothing to Enhance the SAR Image Attack Transferability [ICASSP 2024]**

Yue Xu, Xin Liu, Kun He, Shao Huang, Yaodong Zhao, Jie Gu.

## Requirements
* Python == 3.8.13
* pytorch == 1.9.0+cu102
* torchvision == 0.10.0
* numpy == 1.21.2
* pandas == 2.0.3
* opencv-python == 4.7.0.72

## Dataset

### MSTAR (Moving and Stationary Target Acquisition and Recognition) Database

#### Overview
You should download the data before running the code. Put the data in ./dataset/soc. The data can be downloaded from [here](https://1drv.ms/f/s!AkwsJ37-ugI4gQ8oPIoYImlZ4he3?e=E4CtJT).

- SOC
    - eval_1: 1000 pic choiced from 10 classes SAR test dataset, 100 pic per class. Using for generate adversarial examples and testing the transferability.
    - eval_data_1.pkl: The ".pkl" file of eval_1.

#### Format

The raw data the following format:
- Header
    - Type: ASCII
    - Including data shape(width, height), serial number, azimuth angle, etc.
- Data
    - Type: Two-bytes
    - Shape: W x H x 2
        - Amplitude block
        - Phase Block

The below figure is an example of the amplitude block(Left) and phase block(Right)

![Example of data block: 2S1](./experiments/pic/Ori.pdf)

In this repository, the data is converted to ".npy" format for easy use. 

## Model
You should download the pre-trained model before running the code. Put the data in ./experiments/net. The model can be downloaded from [here](https://1drv.ms/f/s!AkwsJ37-ugI4gXlAOcwO3uF_pSnL?e=keIOvc).

Four models are used in this repository:
- ResNet_18
- ResNet_50
- Inception_v3
- InceptionResNet_v2

You can train other SAR classification models and use them for attacking experiments.

## Quick Start Guide for Attacking and Adversarial Examples Generation

1. You can generate the adversarial examples by running the following command:
```shell
$ cd src 
$ python3 Attack.py
```

2. The generated adversarial examples would be stored as a ".pkl" file in the directory **./dataset/soc/XX**. ("XX" is the model name). The overview image will be stored in the directory **./dataset/soc/per_img**. 

3. You can run the file **'Trans_attack.py'** to evaluate the attack transferability for target models.
```shell
$ cd src 
$ python3 Trans_attack.py
```
4. If you try to attack models you trained by yourself, you should make sure that the original test data you use is correctly classified and generate a new batch of test data.

## Note
The code is used for the IMGS method, if you want to know more details about SAR-ATR model training, you can learn from other projects.

## Citation
If our paper or this code is useful for your research, please cite our paper.
