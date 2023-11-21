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

or
```bash
git clone git@github.com:cmarti/epik.git
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

## Installing and testing KeOps

We have sometimes run into problems when running KeOps library on the GPU. We found two types of issues
To ensure that the pykeops installation worked well before trying out EpiK. Open a python console and run

```python
import pykeops
pykeops.test_torch_bindings()
```

If this runs without errors then you should be good to go

See [pykeops](https://www.kernel-operations.io/keops/python/installation.html) docs for more general information

### Issues with cuda versions

In out HPC environment we often run into problems for loading and matching the CUDA versions that are compatible with the tested version of pykeops v2.1.2.

It will show an error similar to this in that case:

```python
  File "/grid/mccandlish/home_norepl/martigo/miniconda3/envs/epik/lib/python3.8/site-packages/torch/autograd/function.py", line 506, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/grid/mccandlish/home_norepl/martigo/miniconda3/envs/epik/lib/python3.8/site-packages/pykeops/torch/generic/generic_red.py", line 77, in forward
    myconv = keops_binder["nvrtc" if tagCPUGPU else "cpp"](
KeyError: 'nvrtc'
```

We have solved that by installing the same verison of the cudatoolkit as the cuda system.

Check CUDA version by running the following command on the command line. It should generate an output like the following
```bash
nvidia-smi
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.223.02   Driver Version: 470.223.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro P620         Off  | 00000000:01:00.0  On |                  N/A |
| 34%   37C    P8    N/A /  N/A |    933MiB /  1977MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

Then use conda to install the corresponding cudatoolkit that is required by pykeops

```bash
conda install -c conda-forge cudatoolkit=XX.X
```

### Issues with pykeops cache dir

pykeops needs a cache directory to store the compiled code on the fly. This directory is set well automatically always such that it is not accessible to pykeops directly. In that case, it will return an error like the following:

```python
/apps/nlp/1.2/lib/python3.8/ctypes/__init__.py in __init__(self, name, mode, handle, use_errno, use_last_error, winmode)
    371 
    372         if handle is None:
--> 373             self._handle = _dlopen(self._name, mode)
    374         else:
    375             self._handle = handle

OSError: /home/juannanzhou/.cache/keops2.1.2/Linux_c1000a-s17.ufhpc_4.18.0-477.27.1.el8_8.x86_64_p3.8.12_CUDA_VISIBLE_DEVICES_2_3/nvrtc_jit.so: cannot open shared object file: No such file or directory
```

Specifying these cache dir to a known accessible path solved the issue for us.

```python
import pykeops
print(pykeops.get_build_folder())  # display current build_folder
pykeops.set_build_folder("/my/new/location")  # change the build folder
print(pykeops.get_build_folder())  # display new build_folder
```




