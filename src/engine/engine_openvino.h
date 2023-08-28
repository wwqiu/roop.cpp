#include "engine.h"
#include <inference_engine.hpp>

using namespace InferenceEngine;

namespace WW {
class EngineOpenVINO {
public:
	EngineOpenVINO() = default;
	StateCode Load(std::string model_path);
	InputTensorMap& Inputs();
	OutputTensorMap& Outputs();
	StateCode Infer(InputTensorMap& inputs, OutputTensorMap& outputs);

private:
	static Core ie_core_;
	CNNNetwork cnn_network_;
	InferRequest request_;
	ExecutableNetwork network_;
};
}