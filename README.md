# Requirements

First of all, install the protocol buffer library following the [instructions](https://github.com/protocolbuffers/protobuf/tree/master/src)

[GSL](https://www.gnu.org/software/gsl/)
For linux distros it should suffice to run
```shell
  sudo apt-get install libgsl-dev
```

[2to3]()
```shell
  sudo apt-get install 2to3
  sudo apt-get install python3-lib2to3
  sudo apt-get install python3-toolz
```

Although it is possible to use this repository as a standalone C++ project
(see spatial_mix/run_from_file.cpp), it is highly recommended to export 
the C++ executables to Python, for this the pybind11 library is needed.
To install:
```shell
  pip3 install pybind11
```

# Setup

After cloning the repository, update the submodules via

```shell
  git submodule update --recursive
```

Then, from the terminal run
```shell
  cd lib/math
  ./runTests.py test/unit/math/rev/core/agrad_test.cpp
```

To compile the protocol buffers
```shell
  cd spatial_mix
  make compile_protos
```

To compile the C++ code and generate the Python package
```shell
  cd spatial_mix
  make generate_pybind
```
