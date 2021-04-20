#pragma once
#ifndef SRC_SEGMENTATION_EXAMPLE_APP_H_
#define SRC_SEGMENTATION_EXAMPLE_APP_H_

#include <torch/script.h>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <vector>

void calc();
cv::Mat segment(cv::Mat& src_img);
void initializeNeuralNetwork();
#endif // SRC_SEGMENTATION_EXAMPLE_APP_H_