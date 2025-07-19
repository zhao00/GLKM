### Description
This is the official code for the paper **["Efficient Greedy Optimization Method for k-means"](https://authors.elsevier.com/a/1lSX677nKsAg3)**, published in *Pattern Recognition*, 2026.
The core algorithm is implemented in C++ and provides a Python interface.

### Installation
```bash
cd GKM
python setup.py build_ext --inplace
cd ../LKM
python setup.py build_ext --inplace
```

### requirements
```bash
python (>=3.6, <3.12 recommended)
cython
setuptools==57.5.0
```

### Usage
see demo.py


+ **GKM** is the core part of the paper and implements the proposed greedy local k-means algorithm.

+ FeiPub is based on the [FeiPub repository](https://github.com/ShenfeiPei/FeiPub).

+ LKM was written by [shenfeipei](https://github.com/ShenfeiPei) to fune-tune the objective function.

### Cite
```bash
@article{YUAN2026112140,
title = {Efficient greedy optimization method for k-means},
journal = {Pattern Recognition},
volume = {171},
pages = {112140},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112140},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325008003},
author = {Yuan Yuan and Lin Zhao and Shenfei Pei and Feiping Nie},
```
