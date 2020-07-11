#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <random>

#include <iostream>
#include "NvUtils.h"
#include "NvOnnxParser.h"
using namespace nvinfer1;


std::string model_path = "/workspace/tensorrt/data/cifar/cifar.onnx";

int main(int argc, char** argv) {
  auto builder = createInferBuilder(sample::gLogger.getTRTLogger());

  auto config = builder->createBuilderConfig();
  Dims4 dims1(1,10,10,3);
  Dims4 dims2(1,32,32,3);
  Dims4 dims3(1,32,64,3);
  std::string input_name = "x";
  for (int i=0; i<2; ++i){
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input_name.c_str(), OptProfileSelector::kMIN, dims1);
    profile->setDimensions(input_name.c_str(), OptProfileSelector::kOPT, dims2);
    profile->setDimensions(input_name.c_str(), OptProfileSelector::kMAX, dims3);
    config->addOptimizationProfile(profile);
  }

  auto network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
  parser->parseFromFile(model_path.c_str(), 3);
  auto engine = builder->buildEngineWithConfig(*network,*config);

	std::vector<IExecutionContext*> contexts;
	
	        
	sample::gLogInfo << "NbOptimizationProfiles: "<< engine->getNbOptimizationProfiles() << std::endl;
	sample::gLogInfo << "NbEngineBindings: " << engine->getNbBindings() << std::endl;
	sample::gLogInfo << "[" << input_name << "] Input Binding Index: " << engine->getBindingIndex(input_name.c_str()) << std::endl;
	for (int binding=0; binding < engine->getNbBindings(); binding++) {
		sample::gLogInfo << "Binding " << binding << ": " << engine->getBindingName(binding) << std::endl;
	}
	
  for (int i=0; i<2; ++i){
    contexts.emplace_back(engine->createExecutionContext());
    auto context = contexts.back();
    context->setOptimizationProfile(i);
    std::cout<<"allInputDimensionsSpecified: "<<context->allInputDimensionsSpecified()<<"\n";
    context->setBindingDimensions(0, dims2);
    std::cout<<"allInputDimensionsSpecified must equal 1: "<<context->allInputDimensionsSpecified()<<"\n";
  }
}