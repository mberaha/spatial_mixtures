# Setup

First of all, install the protocol buffer libary following the [instructions](https://github.com/protocolbuffers/protobuf/tree/master/src)

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
