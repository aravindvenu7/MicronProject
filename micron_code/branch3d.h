// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_3D_RECONSTRUCTION_BRANCH3D_H_
#define SRC_3D_RECONSTRUCTION_BRANCH3D_H_

#include <vector>

#include <opencv2/opencv.hpp>
#include "points_matching_surf.h"
#include "branch.h"

class Branch3d {
public:
	Branch3d(Branch currentBr, cv::Mat warp_mat1, cv::Mat warp_mat2);
	cv::Mat getCentersCoord3d();
	std::vector<double> getRadii();
	int getVesselNumber();

private:
	std::vector<cv::Point2d> findWarpedCenters(
		cv::Mat warp_mat, std::vector<cv::Point2i> original_centers);
	cv::Point2d findWarpedCoordinates(cv::Mat warp_mat, cv::Point2i i);
	cv::Point2d findWarpedCoordinates(cv::Mat warp_mat, cv::Point2d i);
	cv::Mat find3dPointsDistortionCorrection();
	cv::Mat find3dPoints(std::vector<cv::Point2i> coord_left,
		std::vector<cv::Point2d> coord_right);
	cv::Mat find3dPoints(std::vector<cv::Point2d> coord_left,
		std::vector<cv::Point2d> coord_right);
	std::vector<double> computeRealRadii(
		std::vector<double> radii_2d, std::vector<cv::Point2d> radii_direction_2d,
		std::vector<cv::Point2i> centers_left, cv::Mat centers_3d,
		cv::Mat warp_m);
	std::vector<cv::Point2i> centers_coord_left;
	std::vector<cv::Point2d> centers_coord_right;
	cv::Mat centers_coord_3d;
	std::vector<double> radii;
	int vessel_number;
};

#endif  // SRC_3D_RECONSTRUCTION_BRANCH3D_H_
