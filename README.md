<div align='center'>
<h1>InsCD: A Modularized, Comprehensive and User-Friendly Toolkit for Machine Learning Empowered Cognitive Diagnosis</h1>
    <a href='https://aiedu.ecnu.edu.cn/'>Shanghai Institute of AI Education</a>, <a href='http://www.cs.ecnu.edu.cn/'>School of Computer Science and Technology</a><br>
    East China Normal University

<div>InsCD, namely Instant Cognitive Diagnosis (Chinese: 时诊), is a highly modularized python library for cognitive diagnosis in intelligent education systems. This library incorporates both traditional methods (e.g., solving IRT via statistics) and deep learning-based methods (e.g., modelling students and exercises via graph neural networks). 

## 📰 News 
- [x] [2025.7.10] InsCD toolkit v1.3 is released.
  What's New: We implement one new model: Disentangled Graph Cognitive Diagnosis (DisenGCD)
- [x] [2024.8.31] InsCD toolkit v1.2 is released.
What's New: We implement two new models: symbolic cognitive diagnosis model (SymbolCD) and hypergraph cognitive diagnosis model (HyperCD)
- [x] [2024.7.14] InsCD toolkit v1.1 is released and available for downloading.
- [x] [2024.4.20] InsCD toolkit v1.0 is released.

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
pip install inscd-tookit
```

### Quick Example
The following code is a simple example of cognitive diagnosis implemented by inscd. We load build-in datasets, create cognitive diagnosis model, train model and show its performance:  
```python
inscd_run --model NCDM --datahub_name Math1
```
If you want to use multi-GPU parallel training, please use the following command-line:
```
accelerate launch -m inscd_run --model NCDM --datahub_name Math1
```
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

## 🤔 Frequent Asked Questions
> Why I cannot download the dataset when using build-in datasets class (e.g., `NeurIPS20` in `inscd.datahub`)?

Since these datasets are saved in the  Google Driver, they may be not available in some countries and regions. You can use proxy and add the following commands in your terminal:
```bash
export http_proxy = 'http://<IP address of proxy>:<Port of proxy>'
export https_proxy = 'http://<IP address of proxy>:<Port of proxy>'
export all_proxy = 'socks5://<IP address of proxy>:<Port of proxy>'
```

> 💡 Note: These settings are only effective for the current terminal session.
