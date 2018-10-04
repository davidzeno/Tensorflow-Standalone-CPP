# Compiling Tensorflow C++ API and Building a Standalone C++ Tensorflow Application
Follow this to compile Tensorflow C++ API and build a standalone C++ Tensorflow application.
- Tensorflow  version: 1.11.0-rc1
- OS: Linux (Ubuntu 14.04.5)

## Clone tensorflow repo:
```console
git clone https://github.com/tensorflow/tensorflow.git
```
## Compile Tensorflow libtensorflow_cc.so
```console
cd tensorflow
./configure
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
```
While waiting for tensorflow to compile...

## Install protobuf (v3.6.0)
```console
git clone https://github.com/google/protobuf.git
cd protobuf
git checkout v3.6.0
./autogen.sh
./configure
make
make check
sudo make install
sudo ln -s /usr/local/lib/libprotoc.so.16 /usr/lib/
sudo ln -s /usr/local/lib/libprotobuf.so.16 /usr/lib/
```
## Install Eigen (3.3.0)
```console
wget https://bitbucket.org/eigen/eigen/get/fd6845384b86.tar.gz
tar -xvzf fd6845384b86.tar.gz
cd eigen-eigen-fd6845384b86/
mkdir build
cd build
cmake ..
make
sudo make install                                                                                           ```
```
## Install Absl 
```console
wget https://github.com/abseil/abseil-cpp/archive/f21d187b80e3b7f08fb279775ea9c8b48c636030.tar.gz
tar -xvzf f21d187b80e3b7f08fb279775ea9c8b48c636030.tar.gz
```
## Build lib directory
```console
mkdir lib
cp tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so lib/
cp tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so lib/
cp protobuf/src/libprotobuf.la lib/
```
## Build include directory
```console
mkdir -p include/tensorflow
cp -r tensorflow/bazel-genfiles/* include/
cp -r tensorflow/tensorflow/cc include/tensorflow
cp -r tensorflow/tensorflow/core/ include/tensorflow
cp -r tensorflow/third_party include
cp -r /usr/local/include/google/ include/
cp -r /usr/local/include/eigen3/* include
cp -r abseil-cpp-f21d187b80e3b7f08fb279775ea9c8b48c636030/absl/ include/
```
## Compile your C++ code 
```console
g++ -std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w -o SampleNeuralNet SampleNeuralNet.cpp -Iinclude -lprotobuf -pthread -lpthread -Llib -Wl,-Rlib -ltensorflow_cc -ltensorflow_framework
```
## Sample code
```cpp
#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/version.h"

int main() {
    using namespace std;
    using namespace tensorflow;
    using namespace tensorflow::ops;
    using std::vector;

    cout << "Tensorflow version: " << TF_VERSION_STRING << endl;
    Scope root = Scope::NewRootScope();

    // Original Neural State
    // [10.1 20.2; -30.3 40.4]
    cout << "NeuralState = [10.1 20.2; -30.3 40.4]" << endl;
    auto NeuralState = Const(root, { {10.1f, 20.2f}, {-30.3f, 40.4f}});

    // Input Vector
    //[50.5 60.6]
    cout << "InputVector = [50.5 60.6]" << endl;
    auto InputVector = Const(root, { {50.5f, 60.6f}});

    // Output Vector
    // Matrix v = Ab^T
    cout << "OutputVector = NeuralState * InputVector ^ TransposeB" << endl;
    auto OutputVector = MatMul(root.WithOpName("OutputVector"), NeuralState, InputVector, MatMul::TransposeB(true));

    vector<Tensor> outputs;
    ClientSession session(root);

    // Run and fetch OutputVector
    TF_CHECK_OK(session.Run({OutputVector}, &outputs));

    // Expect outputs[0] == [1734.17; 918.09]
    //LOG(INFO) << outputs[0].matrix<float>();
    cout << "OUT {" << endl << " " << outputs[0].matrix<float>() << endl << "}"<< endl;
    
    return 0;
}
```
## Create a tar.gz from lib/ and include/
```console
tar -zcvf tensorflow-c++-libs-includes-`date +"%m-%d-%y"`.tar.gz lib/ include/
```
## Download it 
<a href="https://github.com/davidzeno/Compile-Tensorflow-CPP-API/releases/download/v1.0/tensorflow-c++-libs-includes-10-04-18.tar.gz">tensorflow-c++-libs-includes-10-04-18.tar.gz</a>
