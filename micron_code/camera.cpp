// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include "camera.h"
#include <ctime>
#include <sstream>
#include <iostream>

// IMAGES has to be 800x608. 
const unsigned int SERIAL_RIGHT = 7510609; //Right ROI offset 232,168
const unsigned int SERIAL_LEFT = 7510524; // Left ROI offset 232, 66
/*const unsigned int LEFTCAM_x = 232;
const unsigned int LEFTCAM_y = 66; //60
const unsigned int RIGHTCAM_x = 232;//0
const unsigned int RIGHTCAM_y = 168;*/
bool firsttime = true;
FlyCapture2::PGRGuid guid_left;
FlyCapture2::PGRGuid guid_right;
FlyCapture2::Error error;
FlyCapture2::Camera camera;


void openCameras() {
	// Initialize BusManager and retrieve number of cameras detected
	FlyCapture2::BusManager busMgr;
	unsigned int numCameras;
	error = busMgr.GetNumOfCameras(&numCameras);

	std::cout << "Number of cameras detected: " << numCameras << std::endl;
	if (numCameras < 2) {
		std::cout << "Insufficient number of cameras." << std::endl;
	}
	error = busMgr.GetCameraFromSerialNumber(SERIAL_LEFT, &guid_left);
	error = busMgr.GetCameraFromSerialNumber(SERIAL_RIGHT, &guid_right);
}

void takeBothPictures(cv::Mat* image_left, cv::Mat* image_right) {
	*image_left = takePicture1(guid_left);
	*image_right = takePicture1(guid_right);
}

cv::Mat takeLeftPicture() {
	return takePicture(guid_left);
}



cv::Mat takePicture(FlyCapture2::PGRGuid& guid)
{
	
	
	
	FlyCapture2::Image rawImage;
	if (firsttime)
	{
		
		error = camera.Connect(&guid);

		// Start streaming on camera
		error = camera.StartCapture();
		// Get the image
	}
	firsttime = false;
	int tk = clock();
	error = camera.RetrieveBuffer(&rawImage);
	

	int tj = clock();
	// convert to rgb
	FlyCapture2::Image rgbImage;
	rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);
	
	//std::cout << "picture till retrieval takes" << (tj - tk) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
	// convert to OpenCV Mat
	unsigned int rowBytes = static_cast<double>(rgbImage.GetReceivedDataSize()) / static_cast<double>(rgbImage.GetRows());
	cv::Mat image = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);
	cv::Mat image2;
	image.copyTo(image2);
	//error = camera.StopCapture();
	return image2;
}

cv::Mat takePicture1(FlyCapture2::PGRGuid& guid) {
	FlyCapture2::Camera camera;
	error = camera.Connect(&guid);
	// Start streaming on camera
	error = camera.StartCapture();
	// Get the image
	FlyCapture2::Image rawImage;
	error = camera.RetrieveBuffer(&rawImage);
	// convert to rgb
	FlyCapture2::Image rgbImage;
	rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);
	// convert to OpenCV Mat
	unsigned int rowBytes = static_cast<double>(rgbImage.GetReceivedDataSize()) / static_cast<double>(rgbImage.GetRows());
	cv::Mat image = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);
	cv::Mat image2;
	image.copyTo(image2);
	error = camera.StopCapture();

	return image2;
}


