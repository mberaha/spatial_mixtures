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
  make compile_protos
```

To run one of the spike tests (test_protos, test_stan, test_pg)
```shell
  make test_protos.out
  ./test_protos.out
```
