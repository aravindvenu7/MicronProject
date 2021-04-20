// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_TIP_POSITION_AND_CONTROL_VISION_AND_CONTROL_H_
#define SRC_TIP_POSITION_AND_CONTROL_VISION_AND_CONTROL_H_

//#define GLOG_NO_ABBREVIATED_SEVERITIES
//#define NO_STRICT

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

void controlMicronWithVision();
cv::Point3d computeGoalPosition(double dist, cv::Point3d versor,
	cv::Point3d tip);
cv::Point3d computeGoalPosition_INSIDE(double closest_radius, double distance_tip_center, cv::Point3d versor,
	cv::Point3d tip);
void waitConnection();


cv::Point3d calcversor(cv::Point3d center, cv::Point3d tip);
double calcdistance(cv::Point3d p1, cv::Point3d p2);
#endif  // SRC_TIP_POSITION_AND_CONTROL_VISION_AND_CONTROL_H_
