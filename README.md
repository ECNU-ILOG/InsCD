<div align='center'>
<h1>InsCD: A Modularized, Comprehensive and User-Friendly Toolkit for Machine Learning Empowered Cognitive Diagnosis</h1>

<a href='https://aiedu.ecnu.edu.cn/'>Shanghai Institute of AI Education</a>, <a href='http://www.cs.ecnu.edu.cn/'>School of Computer Science and Technology</a>

East China Normal University

<img src='asset/inscd.svg' width=700 />
</div>
<br>
<div align='center'>
<a href='https://pypi.org/project/inscd-toolkit/1.3.0/'><img src='https://img.shields.io/badge/pypi-1.3.0-orange'></a> 
<a href=''><img src='https://img.shields.io/badge/Paper-PDF-yellow'></a>

</div>

------

## 🧠 Introduction

**InsCD** (Instant Cognitive Diagnosis, Chinese name：时诊) is a highly modularized Python library for cognitive diagnosis in intelligent education systems.  
It integrates both classical psychometric models (e.g., IRT) and modern deep learning-based approaches (e.g., GNN-based cognitive diagnosis).  
InsCD is designed for extensibility and ease-of-use, enabling researchers and practitioners to quickly evaluate, build, and extend diagnosis models.

------

## 📰 News 
- [x] [2025.7.10] InsCD toolkit v1.3.0 is released.
  What's New: We implement one new model: Disentangled Graph Cognitive Diagnosis (DisenGCD).
- [x] [2024.8.31] InsCD toolkit v1.2 is released.
What's New: We implement two new models: symbolic cognitive diagnosis model (SymbolCD) and hypergraph cognitive diagnosis model (HyperCD)
- [x] [2024.7.14] InsCD toolkit v1.1 is released and available for downloading.
- [x] [2024.4.20] InsCD toolkit v1.0 is released.

------

## 🚀 Getting Started
### Installation
Git and install with pip:
```
git clone https://github.com/ECNU-ILOG/InsCD.git
cd <path of code>
pip install .
```
or install the library from pypi
```
pip install inscd-tookit==1.3.0
```

#### ⚠️ Note: Installing DGL

InsCD depends on **DGL (Deep Graph Library)**. The installation of DGL varies depending on your operating system, PyTorch version, and whether you are using a GPU.

Please refer to the official DGL installation guide and choose the command that matches your environment:

👉 DGL Installation Guide: [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)

### Quick Example

Run the following examples via CLI to quickly train various models. Replace `Math1` with any supported dataset name. 
#### ➤ Classical Models

```bash
inscd_run --model IRT --datahub_name Math1
inscd_run --model MIRT --datahub_name Math1
```

#### ➤ Neural Network-based Models

```bash
inscd_run --model NCDM --datahub_name Math1
inscd_run --model KaNCD --datahub_name Math1
inscd_run --model KSCD --datahub_name Math1
```

#### ➤ Graph-based Models

```bash
inscd_run --model RCD --datahub_name Math1
inscd_run --model SCD --datahub_name Math1
inscd_run --model DisenGCD --datahub_name Math1
inscd_run --model ORCDF --datahub_name Math1
inscd_run --model HyperCD --datahub_name Math1
```

#### ➤ Augmented & Symbolic Models

```bash
inscd_run --model ICDM --datahub_name Math1
inscd_run --model SymbolCD --datahub_name Math1
```

For **multi-GPU training**:

```bash
accelerate launch -m inscd_run --model NCDM --datahub_name Math1
```

------

## 🛠 Implementation
We incoporate classical, famous and state-of-the-art methods published or accepted by leading journals and conferences in the field of psychometric, machine learning and data mining. The reason why we call this toolkit "modulaized" is that we not only provide the "model", but also divide the model into two parts (i.e., extractor and interaction function), which enables us to design new models (e.g., extractor of Hypergraph with interaction function of KaNCD). To evaluate the model, we also provide vairous open-source datasets in online or offline scenarios.

### List of Models
|Model|Release|Paper|
|-----|------------|-----|
|Item Response Theory (IRT)|1952|Frederic Lord. A Theory of Test Scores. _Psychometric Monographs_.|
|Multidimentional Item Response Theory (MIRT)|2009|Mark D. Reckase. _Multidimensional Item Response Theory Models_.|
|Neural Cognitive Diagnosis Model (NCDM)|2020|Fei Wang et al. Neural Cognitive Diagnosis for Intelligent Education Systems. _AAAI'20_.|
|Relation Map-driven Cognitive Diagnosis Model (RCD)|2021|Weibo Gao et al. RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems. _SIGIR'21_.|
|Knowledge-association Neural Cognitive Diagnosis (KaNCD)|2022|Fei Wang et al. NeuralCD: A General Framework for Cognitive Diagnosis. _TKDE_.|
|Knowledge-sensed Cognitive Diagnosis Model (KSCD)|2022|Haiping Ma et al. Knowledge-Sensed Cognitive Diagnosis for Intelligent Education Platforms. _CIKM'22_.|
|Cognitive Diagnosis Model Focusing on Knowledge Concepts (CDMFKC)|2022|Sheng Li et al. Cognitive Diagnosis Focusing on Knowledge Concepts. _CIKM'22_.|
|Self-supervised Cognitive Diagnosis Model (SCD)|2023|Shanshan Wang et al. Self-Supervised Graph Learning for Long-Tailed Cognitive Diagnosis. _AAAI'23_.|
|Disentangled Graph Cognitive Diagnosis (DisenGCD)|2024|Shanshan Wang et al. DisenGCD: A Meta Multigraph-assisted Disentangled Graph Learning Framework for Cognitive Diagnosis. _NeurIPS'24_.|
|Inductive Cognitive Diagnosis  Model (ICDM)|2024|Shuo Liu et al. Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Intelligent Education Systems. _WWW'24_.|
|Symbolic Cognitive Diganosis Model (SymbolCD)|2024|Junhao Shen et al. Symbolic Cognitive Diagnosis via Hybrid Optimization for Intelligent Education Systems. _AAAl'24_.|
|Oversmoothing-Resistant Cognitive Diagnosis Framework (ORCDF)|2024|Shuo Liu et al. ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems. _KDD'24_.|
|Hypergraph Cognitive Diagnosis Model (HyperCDM)|2024|Junhao Shen et al. Capturing Homogeneous Influence among Students: Hypergraph Cognitive Diagnosis for Intelligent Education Systems. _KDD'24._|

### List of Build-in Datasets
|Dataset|Release|Source|
|-------|-------|------|
|`inscd.datahub.Assist17`|2018|https://sites.google.com/view/assistmentsdatamining/dataset|
|`inscd.datahub.FracSub`|2015|http://staff.ustc.edu.cn/%7Eqiliuql/data/math2015.rar|
|`inscd.datahub.Junyi734`|2015|https://www.educationaldatamining.org/EDM2015/proceedings/short532-535.pdf|
|`inscd.datahub.Math1`|2015|http://staff.ustc.edu.cn/%7Eqiliuql/data/math2015.rar|
|`inscd.datahub.Math2`|2015|http://staff.ustc.edu.cn/%7Eqiliuql/data/math2015.rar|
|`inscd.datahub.Matmat`|2019|https://github.com/adaptive-learning/matmat-web|
|`inscd.datahub.NeurIPS20`|2020|https://eedi.com/projects/neurips-education-challenge|
|`inscd.datahub.XES3G5M`|2023|https://github.com/ai4ed/XES3G5M|

Note that we preprocess these datasets and filter invalid response logs. We will continuously update preprocessed datasets to foster the community.

------

## 📦 Requirements

```
gdown==5.2.0
pandas==2.2.3
numpy==1.26.4
torch==2.4.0
scikit-learn==1.5.2
scipy==1.13.1
joblib==1.4.2
tqdm==4.66.5
accelerate==1.1.1
pyyaml==6.0.2
wandb==0.18.5
deap==1.4.1
networkx==3.2.1
```
------

## 🤔 Frequent Asked Questions
> Why I cannot download the dataset when using build-in datasets class (e.g., `NeurIPS20` in `inscd.datahub`)?

Since these datasets are saved in the  Google Driver, they may be not available in some countries and regions. You can use proxy and add the following commands in your terminal:
```bash
export http_proxy = 'http://<IP address of proxy>:<Port of proxy>'
export https_proxy = 'http://<IP address of proxy>:<Port of proxy>'
export all_proxy = 'socks5://<IP address of proxy>:<Port of proxy>'
```

> 💡 Note: These settings are only effective for the current terminal session.

------

## 🤗 Contributor

Contributors are arranged in alphabetical order by first name. We welcome more people to participate in maintenance and improve the community of intelligent education.

()

------

## 🧾 Citation
If this toolkit is helpful and can inspire you in your reseach or applications, please kindly cite as follows.

### BibTex
```

```
### ACM Format
