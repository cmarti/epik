# GPU-accelerated Gaussian Process regression in sequence space


## Installation

Create a python3 conda environment and activate it

```bash
conda create -n epik python=3.7
conda activate epik
```

Download the repository using git and cd into it

```bash
git clone git@bitbucket.org:cmartiga/epik.git
```

Install using setuptools
```bash
cd epik
pip install -r requirements.txt
python setup.py install
```

Test installation

```bash
python -m unittest epik/test/*py
```
