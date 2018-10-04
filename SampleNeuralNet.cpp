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

