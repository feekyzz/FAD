
[![Coverage Status](https://coveralls.io/repos/github/neurodata/mgcpy/badge.svg?branch=master)](https://coveralls.io/github/neurodata/mgcpy?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)


`FAD` is a object detection network.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Dataset](#Dataset Link)
- [License](#license)

# Overview
``FAD`` aims to achieve efficient detection of traffic scene objects. Given the pressing demand for traffic object detection in autonomous driving vehicles, attaining efficiency and accuracy in perception on in-vehicle platforms remains a formidable challenge. To this end, we propose a frequency-oriented adaptive detector (FAD) for vehicle-mounted intelligent traffic object detection. FAD effectively integrates high- and low-frequency information to enhance the texture and semantic features of multi-scale objects in complex traffic scenarios, all while maintaining minimal computational overhead. Initially, we present a frequency dynamic convolution (FDC) for a lightweight backbone, aiming at frequency-domain adaptive dilated receptive fields and balancing effective bandwidth. Wherein, the adaptive kernel decomposes the convolution weights into high- and low-frequency components. Subsequent to frequency-domain selection, the convolutional weights are re-calibrated to equilibrate the frequency-domain components within the feature map. In addition, we propose an adaptive frequency-oriented fusion (AFF) framework, which adaptively reorganizes high and low-frequency features across different scales. AFF effectively balances object details, fine boundaries and deep semantic features, aiming to bridge the gap caused by feature inconsistencies.


# System Requirements
## Hardware requirements
`FAD`  is conducted in the PyTorch framework on a Windows system, utilizing a workstation equipped with an RTX 4070 Ti 12-GB GPU and 32-GB RAM.

## Software requirements
### OS Requirements
This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:
+ Windows 10
+ Linux: Ubuntu 16.04

### Python Dependencies
`FAD` mainly depends on the Python scientific stack.

```
basicsr==1.4.2
beautifulsoup4==4.13.4
einops==0.8.1
grad_cam==1.5.4
ipython==8.12.3
matplotlib==3.5.3
mmcv_full==1.7.2
numpy==1.21.6
onnx==1.14.1
onnxruntime==1.19.2
opencv_contrib_python==4.7.0.72
opencv_python==4.9.0.80
openvino==2023.2.0
pandas==1.3.5
Pillow==9.5.0
Pillow==11.3.0
protobuf==6.31.1
psutil==5.9.8
py_cpuinfo==9.0.0
pycocotools==2.0.6
PyWavelets==1.4.1
PyYAML==6.0.1
PyYAML==6.0.2
scipy==1.7.3
seaborn==0.13.2
sentry_sdk==1.39.1
setuptools==60.2.0
setuptools==65.6.3
thop==0.1.1.post2209072238
timm==1.0.17
torch==1.9.0+cu111
torch_dct==0.1.6
torchsummary==1.5.1
torchvision==0.10.0+cu111
tqdm==4.65.2
torch==1.9.0+cu111

```



# Installation Guide:


## Install from Github
```
git clone https://github.com/feekyzz/FAD
cd FAD
conda create -n FAD python=3.8 anaconda
conda activate FAD
pip install -r requirements.txt
```
- `sudo`, if required

# Dataset

## KITTI:
You can download **KITTI** from the following links:

* Google drive

Link: https://drive.google.com/drive/folders/1Ha6b9a1ri0VtZ4t5QCSMPaFHNWBScAC2?usp=sharing

## BDD100K:
You can download **KITTI** from the following links:

* Google drive

Link: https://drive.google.com/drive/folders/1-gxQox4in4zzxcOgo66xN3BPN7fJbdIr?usp=sharing

## Cityscapes:
You can download **KITTI** from the following links:

* Google drive

Link: https://drive.google.com/file/d/14wKLxFjgBVAdS4U_k0PKynrfJiLKbNdO/view?usp=sharing

# Weights
* Google drive

Link: https://drive.google.com/drive/folders/1bgHoK5ASSymYoppbXnHdhND8rRYUTnpB?usp=sharing

# License

This project is covered under the **Apache 2.0 License**.
