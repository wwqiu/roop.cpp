#include "engine_openvino.h"

namespace WW {

StateCode EngineOpenVINO::Load(std::string model_path) {
    cnn_network_ = ie_core_.ReadNetwork(model_path);
    // auto input_info = cnn_network.getInputsInfo();
    lowLatency2(cnn_network_);
    std::map<std::string, std::string> config = { {PluginConfigParams::KEY_CPU_THREADS_NUM, "8"}};
    network_ = ie_core_.LoadNetwork(cnn_network_, "CPU", config);
    request_ = network_.CreateInferRequest();    
}


StateCode EngineOpenVINO::Infer(InputTensorMap& inputs, OutputTensorMap& outputs) {
    InputsDataMap inputs_info = cnn_network_.getInputsInfo();
    for (auto info : inputs_info) {
        Blob::Ptr input_blob = request_.GetBlob(info.first);
        if (inputs.count(info.first) == 0) {
            fprintf(stderr, "unknown input : %s", info.first.c_str());
            return StateCode::FAILED;
        }
        LockedMemory<void> input_mapped = as<MemoryBlob>(input_blob)->wmap();
        float* blob_data = input_mapped.as<float*>();
        if (blob_data == NULL) {
            fprintf(stderr, "Get input blob failed");
            return StateCode::FAILED;
        }
        const SizeVector& dims = info.second->getTensorDesc().getDims();
        Tensor input = inputs[info.first];
        memcpy(blob_data, input.data, sizeof(float) * input.ElementCount());
    }
    request_.Infer();

}

}