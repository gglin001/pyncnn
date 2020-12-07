# pyncnn
python wrapper of [ncnn](https://github.com/Tencent/ncnn) with [pybind11](https://github.com/pybind/pybind11), only support python3.x now.

## APPEND

本版本基于修改版的 ncnn(https://github.com/gglin001/ncnn), 将 ncnn 中各 layer 都做了对应的 python 映射

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 3.4

**On Windows**

* Visual Studio 2015
* CMake >= 3.4

## Build
1. clone [ncnn](https://github.com/Tencent/ncnn) and [pybind11](https://github.com/pybind/pybind11), build and install with default setting, if you change the install directory, change the cmake commond with your setting.
```bash
cd /path/to/pyncnn
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/ncnn/build/install/lib/cmake/ncnn/ ..
make
```

## Install
```bash
cd /path/to/pyncnn/python
pip install .
```

if you use conda or miniconda, you can also install as following:
```bash
cd /path/to/pyncnn/python
python3 setup.py install
```

## Tests
**test**
```bash
cd /path/to/pyncnn/tests
python3 test.py
```

**benchmark**
```bash
cd /path/to/pyncnn/tests
python3 benchmark.py
```

**benchmark gpu(build ncnn with vulkan)**
```bash
cd /path/to/pyncnn/tests
python3 benchmark_gpu.py
```

## numpy
**ncnn.Mat->numpy.array, with no memory copy**
```bash
mat = ncnn.Mat(...)
mat_np = np.array(mat)
```

**numpy.array->ncnn.Mat, with no memory copy**
```bash
mat_np = np.array(...)
mat = ncnn.Mat(mat_np)
```

## model zoo
install requirements
```bash
pip install tqdm requests portalocker opencv-python
```
then you can import ncnn.model_zoo and get model list as follow:
```bash
import ncnn
import ncnn.model_zoo as model_zoo

print(model_zoo.get_model_list())
```
all model in model zoo has example in examples folder
