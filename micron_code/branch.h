// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_VESSEL_GRAPH_BRANCH_H_
#define SRC_VESSEL_GRAPH_BRANCH_H_

#include <list>
#include <vector>

#include <opencv2/opencv.hpp>

class Branch {
public:
	Branch(int row_start, int col_start, int n, cv::Mat skeleton,
		cv::Mat d_tr_img);
	void exploreBranch(int row, int col);
	std::vector<cv::Point2i> getCentersCoord();
	std::list<cv::Point2i> getNewCentersCoord();
	std::vector<double> getRadii();
	std::vector<cv::Point2d> getRadiiDirection();
	cv::Point2i getStartingPoint();
	cv::Point2i getEndingingPoint();
	cv::Mat getSkeletonColored();
	int getBranchNumber();

private:
	bool isSameBranch(int r1, int c1, std::list<cv::Point2i> pixels);
	int elSameBranch(int r1, int c1, std::list<cv::Point2i> pixels);
	cv::Point2d normalToCurve(cv::Point2d pre_center, cv::Point2d post_center);
	void findAllRadiiDirection();
	std::vector<cv::Point2i> centers_coord;
	std::list<cv::Point2i> new_centers_coord;
	std::vector<double> radii;
	std::vector<cv::Point2d> radii_direction;
	cv::Mat skeleton_colored;
	cv::Mat distances_transformated_img;
	int vessel_number;
};

#endif  // SRC_VESSEL_GRAPH_BRANCH_H_