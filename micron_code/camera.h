// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_CAMERA_CAMERA_H_
#define SRC_CAMERA_CAMERA_H_

#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void openCameras();
void takeBothPictures(cv::Mat* image_left, cv::Mat* image_right);
cv::Mat takeLeftPicture();
cv::Mat takePicture(FlyCapture2::PGRGuid& guid);
cv::Mat takePicture1(FlyCapture2::PGRGuid& guid);

#endif // SRC_CAMERA_CAMERA_H_
