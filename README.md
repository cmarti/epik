# Epistatic Kernels for GPU-accelerated Gaussian process regression

`EpiK` is a Python library designed to infer sequence-function relationships using Gaussian process models. Built on top of [GPyTorch](https://docs.gpytorch.ai) and [KeOps](https://www.kernel-operations.io), `EpiK` enables fitting these models to large datasets containing hundreds of thousands to millions of sequence measurements.

You can find more detailed documentation and tutorials [here](https://epik.readthedocs.io)

Scripts to reproduce the analyses and figures from the paper are avaiable in a separate [repository](https://github.com/cmarti/epik_analyses)


- Juannan Zhou, Carlos Martí-Gómez, Samantha Petti, David M. McCandlish. 
  Learning sequence-function relationships with scalable, interpretable Gaussian processes (2025)
  In preparation.

## Installation

We recommend using an new independent environment with python3.8, as used during 
development and testing of `EpiK` to minimize problems with dependencies. Create a python3 conda environment and activate it

```bash
conda create -n epik python=3.8
conda activate epik
```

### Users

Install with `pip`

```bash
pip install epik
```

### Developers

Download the repository using git and cd into it

```bash
git clone git@github.com:cmarti/epik.git
```

Install repository

```bash
cd epik
pip install .
```

Run tests with `pytest`

```bash
pytest test
```







