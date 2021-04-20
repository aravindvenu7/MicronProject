// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>
#ifndef SRC_TIP_POSITION_ESTIMATE_DISTANCE_TIP_STRUCTURE_H_
#define SRC_TIP_POSITION_ESTIMATE_DISTANCE_TIP_STRUCTURE_H_

#include <list>
#include <opencv2/opencv.hpp>

void estimateTipDistanceAndNormal(cv::Point3d tip_position, cv::Mat all_centers,
	std::list<cv::Point3d> all_points,	double* min_distance,
	cv::Point3d* versor_point_to_tip);
void estimateTipDistanceNormalAndClosest(cv::Point3d tip_position, cv::Mat all_centers,
	std::list<cv::Point3d> all_points, double* distance_tip_cloud,
	cv::Point3d* versor_point_to_tip, cv::Point3d* closest, cv::Point3d* center);
double estimateTipDistance(cv::Point3d tip_position, cv::Mat all_centers,
	std::list<cv::Point3d> all_points);
int findClosestCenterIndex(cv::Point3d tip_position, cv::Mat all_centers);
void estimateVersorDistanceVessel(cv::Point3d tip_position,cv::Point3d deformable_point, double* distance_tip_cloud, cv::Point3d* versor_point_to_tip);
#endif  // SRC_TIP_POSITION_ESTIMATE_DISTANCE_TIP_STRUCTURE_H_