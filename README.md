______________________________________________________________________

<div align="center">

# Depth Estimation (BiFPN + EfficientNet)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

<img src="https://github.com/kzaleskaa/depth-estimation-with-compression/assets/62251989/747c72d8-e096-4113-9951-5886213187bc" />

## Description

This project entails the development and optimization of a depth estimation model based on a UNET architecture enhanced with **Bi-directional Feature Pyramid Network** (BIFPN) and **EfficientNet** components. The model is trained on the NYU Depth V2 dataset and evaluated on the Structural Similarity Index (SSIM) metric.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/kzaleskaa/depth-estimation-with-compression
cd depth-estimation-with-compression

# [OPTIONAL] create conda environment
conda create -n myenv python=3.11
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/kzaleskaa/depth-estimation-with-compression
cd depth-estimation-with-compression

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```


## Results for BiFPN + FFNet

The base model was trained for 25 epochs. QAT was performed for 10 epochs.

**Baseline and Fuse**

<div align=center>

| Method       | test/ssim (Per tensor) | model size (MB) (Per tensor) |
|--------------|------------------------|------------------------------|
| **baseline** | 0.778                  | 3.53                         |
| **fuse**     | 0.778                  | 3.45                         |

</div>


**PTQ, QAT, and PTQ + QAT (Per tensor and Per channel)**

<div align=center>


| Method        | test/ssim (Per tensor)      | model size (MB) (Per tensor)   | test/ssim (Per channel)       | model size (MB) (Per channel)    |
|---------------|-----------------------------|--------------------------------|-------------------------------|----------------------------------|
| **ptq**       | 0.6480          | 0.96791             | 0.6518            | 0.9679               |
| **qat**       | 0.7715          | 0.96791             | 0.7627            | 0.9681               |
| **ptq + qat** | 0.7724          | 0.96899             | 0.7626            | 0.9692               |

</div>