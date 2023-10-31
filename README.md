<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>ADS-B_Classification</h1>
<h3>Jet Kwok (2023/10/31)</h3>

<p align="center">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat-square&logo=YAML&logoColor=white" alt="YAML" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Pytorch-6EFF3.svg?style=flat-square&logo=Pytorch&logoColor=white" alt="JSON" />
</p>
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📦 Features](#-features)
- [📂 repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running ADS-B_Classification](#-running-ADS-B_Classification)
    - [🧪 Tests](#-tests)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

ADS-B Signal Classification Based on Deep Learning.

---

## 📦 Features

The Project contains the following modules:
1. train model
2. test model
3. deploy model


---


## 📂 Repository Structure

```sh
└── ADS-B_Classification/
    ├── __asserts__/
    │   └── figs/
    ├── configs/
    │   └── config.yaml
    ├── data_loaders/
    │   └── my_ADSB_dl.py
    ├── deploy/
    │   ├── STEP01_torch2onnx.py
    │   ├── STEP02_onnx_simplify.py
    │   ├── STEP03_onnx_eval.py
    │   └── STEP04_onnx2trt.py
    ├── experiments/
    │   └── ADS-B/
    │       ├── checkpoints/
    │       ├── deploy/
    │       ├── images/
    │       └── logs/
    ├── infers/
    │   └── ADSB_infer.py
    ├── main_test.py
    ├── main_train.py
    ├── models/
    │   ├── ADSB_model.py
    │   ├── base_model/
    │   │   └── resnet1d.py
    │   └── LSTM.py
    ├── trainers/
    │   └── ADSB_trainer.py
    └── utils/
        ├── loss_utils.py
        ├── utils.py
        ├── utils_common.py
        └── utils_visualize.py

```

---


## ⚙️ Modules

<details closed><summary>Root</summary>

| File                              | Summary                   |
| ---                               | ---                       |
| [main_test.py]({file})            | Project test entrance     |
| [main_train.py]({file})           | Project train entrance    |
| [config.yaml]({file})             | Config file               |
| [my_ADSB_dl.py]({file})           | Customized dataloader     |
| [STEP01_torch2onnx.py]({file})    | PyTorch checkpoint to ONNX file   |
| [STEP02_onnx_simplify.py]({file}) | ONNX file simpl;ify               |
| [STEP03_onnx_eval.py]({file})     | ONNX file and PyTorch checkpoint evaluation                |
| [STEP04_onnx2trt.py]({file})      | ONNX file to TensorRT (TODO) |
| [ADSB_infer.py]({file})           | Main infer|
| [ADSB_model.py]({file})           | Porject model |
| [ADSB_trainer.py]({file})         | Main trainer|
| [loss_utils.py]({file})           | Loss function utils|
| [utils.py]({file})                | Other utils|
| [utils_common.py]({file})         | Common utils|
| [utils_visualize.py]({file})      | Visualize utils|

</details>

---

## 🚀 Getting Started

***Dependencies***


Download motion modules and put them under `ADS-B_Classification/data/`.

- ADS-B Datasets: [Google Drive](https://drive.google.com/file/d/1_suvE5MVsEp3BWEsA8JTlbNF_AZQbhni/view?usp=drive_link) 

### 🔧 Installation

1. Clone the ADS-B_Classification repository:
```sh
git clone ../ADS-B_Classification
```

2. Change to the project directory:
```sh
cd ADS-B_Classification
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🤖 Running ADS-B_Classification

```sh
customize config.yaml
```

### 🚀 Train
```sh
python main_train.py
```

### 🧪 Tests
```sh
python main_test.py
```

---

## 📄 License


This project is protected under the [MIT](https://github.com/FMVPJet/ssps_ubss/blob/main/LICENSE) License. For more details, refer to the [LICENSE](https://github.com/FMVPJet/ssps_ubss/blob/main/LICENSE) file.

---

## 👏 Acknowledgments

- *[JetKwok](https://fmvpjet.github.io/)*

[**Return**](#Top)

---

