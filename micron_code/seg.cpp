
#include <torch/script.h> // One-stop header.
#include "seg.h"
#include <iostream>
#include <memory>
#include <stdio.h>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <vector>

double avgt = 0;
torch::Device device(torch::kCUDA);
torch::jit::script::Module module;
//std::vector<torch::jit::IValue> inputs;

void initializeNeuralNetwork()
{
	at::globalContext().setBenchmarkCuDNN(true);
	//at::globalContext().setBenchmarkCuDNN(true);
	//module->to(torch::Device(torch::kCUDA, 0));
	torch::NoGradGuard no_grad_guard;
	//std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("C:/Users/Aravind/traced_fscnn_model16.pt");
	module = torch::jit::load("D:/Downloads/traced_fscnn_model40.pt");
	module.to(device);

}

cv::Mat segment(cv::Mat& src_img)

{   
	//torch::NoGradGuard no_grad_guard;
	at::globalContext().setBenchmarkCuDNN(true);
	//at::globalContext().setBenchmarkCuDNN(true);
	torch::NoGradGuard no_grad_guard;
	cv::Mat resultImg(224, 288, CV_8UC1);
	torch::Tensor out;
	cv::Mat img_float;
	std::vector<torch::jit::IValue> inputs;
	//torch::jit::script::Module module;
	//module = torch::jit::load("C:/Users/avenugo2/Downloads/traced_fscnn_model16.pt");
	int start = clock();
	
	cv::Size original_size = cv::Size(800, 608);	
	cv::Size network_size = cv::Size(288, 224);	
	cv::resize(src_img, src_img, network_size);
	cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);	
	src_img.convertTo(src_img, CV_32FC3, 1 / 255.0);

	at::Tensor tensorImage = torch::from_blob(src_img.data, { 1, src_img.rows, src_img.cols, 3 }, torch::kFloat).to(device); //at::Tensor vs torch::Tensor
	tensorImage = tensorImage.permute({ 0, 3, 1, 2 });

	//tensorImage.to(device);	
	//inputs.push_back(tensorImage); 
	//at::Tensor output = module.forward().toTensor().detach();
	
	at::Tensor output = torch::argmax(module.forward({ tensorImage }).toTensor().detach(), 1);
	at::Tensor out_tensor = output.permute({1,2,0 });	
	out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
	out_tensor = out_tensor.to(torch::kCPU);	
	std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8)*out_tensor.numel());
	
	cv::resize(resultImg, resultImg, original_size);
	int end = clock();
	//std::cout << "total time for seg" << (end - start) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
	return resultImg;

}

